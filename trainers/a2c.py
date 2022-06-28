from zmq import device
from model import Actor, Critic
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class A2cAgent:
    def __init__(self, observation_spec, action_spec, *args, **kwargs):
        self.observation_spec = observation_spec
        self.action_spec = action_spec

        self.actor = Actor(observation_spec, action_spec)
        self.actor.to(device)

        self.critic = Critic(observation_spec, action_spec)

        lr = 0.0001
        gamma = 0.99
        self.optimizer = optim.Adam(lr=lr, params=self.actor.parameters())
        self.buffer = None

    def step(self, state):
        dist = self.actor.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, torch.sum(entropy, dim=1)

    def add_to_buffer(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def train(self):
        ## train a2c agent

        # sample from buffer
        state, action, reward, next_state, done = self.buffer.sample(batch_size=32)

        # update target networks
        pass

