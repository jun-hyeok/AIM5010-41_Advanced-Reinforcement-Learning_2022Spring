import os.path
import gym
import numpy as np
import pandas as pd
import tensorflow as tf
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)  # type: ignore

from environment import UnityToGymWrapper
from utils.config import ConfigShell

CONFIG_FILE = "config/preset/example.yaml"
PROJECT_HOME = os.path.dirname(__file__)


def load_config(file_name: str) -> dict:
    configsh = ConfigShell(file_name)
    config = pd.json_normalize(configsh.parameter).to_dict("records")[0]
    return config


def make_env(config: dict) -> gym.Env:
    # base_port = int(input("Enter base port: "))
    time_scale = int(config.get("setting.time_scale"))
    width = int(config.get("setting.width"))
    height = int(config.get("setting.height"))
    # env_file = str(config.get("env.path"))
    env_file = "/home/junhyeok/Documents/AIM5010-41_Advanced-Reinforcement-Learning_2022Spring/envs/PushBlockCollab/build"
    channel_config = EngineConfigurationChannel()

    env = UnityEnvironment(
        file_name=env_file,
        # base_port=base_port,
        # no_graphics=True,
        side_channels=[channel_config],
    )
    channel_config.set_configuration_parameters(
        time_scale=time_scale, quality_level=1, width=width, height=height
    )
    env = UnityToGymWrapper(
        env, uint8_visual=True, flatten_branched=True, allow_multiple_obs=True
    )
    return env


if __name__ == "__main__":
    config = load_config(CONFIG_FILE)
    env = make_env(config)

    action_space = env.action_space
    # [5] Training
    n_episode = int(config.get("param.n_episode", 2000))
    global_step = 0

    print("Training...")
    for episode in range(n_episode):
        time_step = 0
        observations = env.reset()
        n_agents = env.n_agents
        done = False
        while not done:
            time_step += 1
            global_step += 1
            debug_log = f"Episode: {episode}/{n_episode} "
            debug_log += f"Global step: {global_step} "
            debug_log += f"Time step: {time_step} "
            action = np.random.randint(action_space.n, size=n_agents)
            next_observations, reward, done, info = env.step(action)

            observations = next_observations.copy()

        if done:
            print("Episode {} finished after {} time steps".format(episode, time_step))

    env.close()
    print("Training finished")

###
"""
observation_space = env.observation_space # type: Box 
env.observation_space.spaces -> [env.observation_space.spaces[0] # .shape -> (20,20,6), env.observation_space.spaces[1] # .shape -> (10,)]

action_space = env.action_space # type: Discrete -> Descret(7) 
action_space.n -> 7

# [1] reset environment, initialize params -> env.reset()
state = env.reset()

n_episode = 0
global_step = 0

for episode in range(n_episode):
    done = False
    time_step = 0

    # [2] while not done, training
    while not done:
        time_step += 1 
        global_step += 1

        # [3] agent action
        action = agent.step(state)

        # policy = agent.behavior_spec[brain_name]
        # [4] state from environment -> env.step(action)
        state, reward, done, info = env.step(action)

        # [5] add trajectory to agent replay buffer -> agent.add_experience(...)
        agent.add_experience(state, action, reward, done)

        # [6] if train, train agent -> agent.train()
        if time_step > agent.train_start:
            agent.train() -> ppo, sac, poca
        if train_model:
            self._save_models()
env.close()
"""

"""
agent.__init__() <- model: ppo, sac, poca
agent.step(state) <- agent.model.forward(state) or random action
agent.add_experience(state, action, reward, done) <- agent.replay_buffer.add(...)
agent.train() <- agent.optimizer.step(...), agent.model.update(...)

env.reset()
env.step(action)
"""
