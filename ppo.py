from trainers.model import Actor, Critic
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PpoAgent:
    clip_norm = 0.2
    max_grad_norm = 0.5
    epoch = 10
    buffer_size = 10000
    batch_size = 8
    actor_lr = 0.0001
    critic_lr = 0.001
    gamma = 0.99

    def __init__(self, observation_spec, action_spec, *args, **kwargs):
        self.observation_spec = observation_spec
        self.action_spec = action_spec

        self.actor = Actor(observation_spec, action_spec).float()
        self.critic = Critic(observation_spec, action_spec).float()
        self.buffer = deque(maxlen=self.buffer_size)
        self.counter = 0
        self.training_step = 0

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def step(self, state):
        dist = self.actor.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, torch.sum(entropy, dim=1)

    def add_to_buffer(self, state, action, reward, next_state, done):
        self.buffer.add([state, action, reward, next_state, done])
        self.counter += 1
        return self.counter % self.batch_size == 0

    def train(self):
        self.training_step += 1

        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.float).view(
            -1, 1
        )
        reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(
            -1, 1
        )
        next_state = torch.tensor(
            [t.next_state for t in self.buffer], dtype=torch.float
        )
        old_action_log_prob = torch.tensor(
            [t.a_log_prob for t in self.buffer], dtype=torch.float
        ).view(-1, 1)

        reward = (reward - reward.mean()) / (reward.std() + 1e-10)
        with torch.no_grad():
            target_v = reward + self.gamma * self.critic(next_state)

        advantage = (target_v - self.critic(state)).detach()
        for _ in range(self.epoch):
            for index in BatchSampler(
                SubsetRandomSampler(range(self.buffer_size), self.batch_size, True)
            ):
                mu, sigma = self.actor_net(state[index])
                n = Normal(mu, sigma)
                action_log_prob = n.log_prob(action[index])
                ratio = torch.exp(action_log_prob - old_action_log_prob)

                L1 = ratio * advantage[index]
                L2 = (
                    torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
                    * advantage[index]
                )
                action_loss = -torch.min(L1, L2).mean()  # MAX->MIN desent
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_net.parameters(), self.max_grad_norm
                )
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(
                    self.critic_net(state[index]), target_v[index]
                )
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic_net.parameters(), self.max_grad_norm
                )
                self.critic_net_optimizer.step()
        del self.buffer[:]


if __name__ == "__main__":
    from common import load_config, make_env

    CONFIG_FILE = "/home/junhyeok/Documents/AIM5010-41_Advanced-Reinforcement-Learning_2022Spring/config/preset/ppo.yaml"
    config = load_config(CONFIG_FILE)
    env = make_env(config)

    observation_space = env.observation_space.spaces
    action_space = env.action_space

    agent = PpoAgent(observation_space, action_space)
    n_episode = 10000
    global_step = 0
    running_reward = -1000
    for episode in range(n_episode):
        state = env.reset()
        done = False
        time_step = 0
        score = 0
        while not done:
            time_step += 1
            global_step += 1
            action, *_ = agent.step(state)
            next_state, reward, done, info = env.step(action)
            if agent.add_to_buffer(state, action, reward, done, next_state):
                agent.train()
            score += reward
            state = next_state.copy()
            running_reward = running_reward * 0.9 + score * 0.1
            if episode % 10 == 0:
                print(
                    f"episode: {episode}/{n_episode}, score: {score}, running_reward: {running_reward}, time_step: {time_step}"
                )
            if running_reward > -200:
                print("Solved! Running reward is now {}!".format(running_reward))
                done = True
    env.close()
    print("Done")

