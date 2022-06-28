import gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                extractors[key] = nn.Sequential(
                    nn.Conv2d(subspace.shape[0], 24, kernel_size=3),  # type: ignore [N, 6, 20, 20] -> [N, 24, 18, 18]
                    nn.ReLU(),
                    nn.Conv2d(24, 24, kernel_size=3),  # type: ignore [N, 24, 18, 18] -> [N, 24, 16, 16]
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),  #  [N, 24, 16, 16] -> [N, 24, 8, 8]
                    nn.Flatten(),  # [N, 24, 8, 8] -> [N, 24 * 8 * 8]
                    # nn.Linear(24 * 8 * 8, 128),  # type: ignore [N, 24, 8, 8] -> [N, 128]
                )
                total_concat_size += 24 * 8 * 8
            elif key == "vector":
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 128),  # type: ignore [N, 10] -> [N, 128]
                    nn.ReLU(),
                )
                total_concat_size += 128

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


if __name__ == "__main__":
    from stable_baselines3.common.policies import MultiInputActorCriticPolicy

    from common import load_config, make_env
    from stable_baselines3 import PPO

    CONFIG_FILE = "config/preset/example.yaml"

    config = load_config(CONFIG_FILE)
    env = make_env(config)
    CustomCombinedExtractor
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        policy_kwargs={"features_extractor_class": CustomCombinedExtractor},
    )
    obs = env.reset()
    n_episode = int(config.get("param.n_episode", 2000))
    model.learn(total_timesteps=n_episode)
    print("Training...")
    # action_space = env.action_space
    global_step = 0
    for episode in range(n_episode):
        time_step = 0
        obs = env.reset()
        n_agents = env.n_agents
        done = False
        while not done:
            time_step += 1
            global_step += 1
            debug_log = f"Episode: {episode}/{n_episode} "
            debug_log += f"Global step: {global_step} "
            debug_log += f"Time step: {time_step} "
            # action = np.random.randint(action_space.n, size=n_agents)
            action, _states = model.predict(obs, deterministic=True)
            actions = [action]
            action, _states = model.predict(obs, deterministic=True)
            actions.append(action)
            action, _states = model.predict(obs, deterministic=True)
            actions.append(action)
            next_obs, reward, done, info = env.step(actions)
            obs = next_obs.copy()
            model.train()
            if done:
                print(
                    "Episode {} finished after {} time steps".format(episode, time_step)
                )
    env.close()
    print("Training finished")
