"https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/runner.py#L38"
import tensorflow as tf
import numpy as np
from gym.spaces import Discrete, Box, MultiDiscrete

# fmt: off
from baselines.common.policies import (PolicyWithValue,
                                       _normalize_clip_observation,
                                       encode_observation, get_network_builder,
                                       observation_placeholder) 
# fmt: on
def observation_placeholder(ob_space, batch_size=None, name="Ob"):
    """
    Create placeholder to feed observations into of the size appropriate to the observation space
    Parameters:
    ----------
    ob_space: gym.Space     observation space
    batch_size: int         size of the batch to be fed into input. Can be left None in most cases.
    name: str               name of the placeholder
    Returns:
    -------
    tensorflow placeholder tensor
    """

    assert (
        isinstance(ob_space, Discrete)
        or isinstance(ob_space, Box)
        or isinstance(ob_space, MultiDiscrete)
    ), "Can only deal with Discrete and Box observation spaces for now"

    dtype = ob_space.dtype
    if dtype == np.int8:
        dtype = np.uint8

    return tf.placeholder(shape=(batch_size,) + ob_space.shape, dtype=dtype, name=name)


def encode_observation(ob_space, placeholder):
    """
    Encode input in the way that is appropriate to the observation space
    Parameters:
    ----------
    ob_space: gym.Space             observation space
    placeholder: tf.placeholder     observation input placeholder
    """
    if isinstance(ob_space, Discrete):
        return tf.to_float(tf.one_hot(placeholder, ob_space.n))
    elif isinstance(ob_space, Box):
        return tf.to_float(placeholder)
    elif isinstance(ob_space, MultiDiscrete):
        placeholder = tf.cast(placeholder, tf.int32)
        one_hots = [
            tf.to_float(tf.one_hot(placeholder[..., i], ob_space.nvec[i]))
            for i in range(placeholder.shape[-1])
        ]
        return tf.concat(one_hots, axis=-1)
    else:
        raise NotImplementedError


def build_policy(
    env,
    policy_network,
    value_network=None,
    normalize_observations=False,
    estimate_q=False,
    **policy_kwargs
):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None):
        ob_space = env.observation_space

        X = (
            observ_placeholder
            if observ_placeholder is not None
            else observation_placeholder(ob_space, batch_size=nbatch)
        )

        extra_tensors = {}

        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors["rms"] = rms
        else:
            encoded_x = X

        encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope("pi", reuse=tf.AUTO_REUSE):
            policy_latent = policy_network(encoded_x)
            if isinstance(policy_latent, tuple):
                policy_latent, recurrent_tensors = policy_latent

                if recurrent_tensors is not None:
                    # recurrent architecture, need a few more steps
                    nenv = nbatch // nsteps
                    assert (
                        nenv > 0
                    ), "Bad input for recurrent policy: batch size {} smaller than nsteps {}".format(
                        nbatch, nsteps
                    )
                    policy_latent, recurrent_tensors = policy_network(encoded_x, nenv)
                    extra_tensors.update(recurrent_tensors)

        _v_net = value_network

        if _v_net is None or _v_net == "shared":
            vf_latent = policy_latent
        else:
            if _v_net == "copy":
                _v_net = policy_network
            else:
                assert callable(_v_net)

            with tf.variable_scope("vf", reuse=tf.AUTO_REUSE):
                # TODO recurrent architectures are not supported with value_network=copy yet
                vf_latent = _v_net(encoded_x)

        policy = PolicyWithValue(
            env=env,
            observations=X,
            latent=policy_latent,
            vf_latent=vf_latent,
            sess=sess,
            estimate_q=estimate_q,
            **extra_tensors
        )
        return policy

    return policy_fn


def net():
    def network_fn(X, nenv=1):
        # return h, {'S':S, 'M':M, 'state':snew, 'initial_state':initial_state}
        return None

    return network_fn
