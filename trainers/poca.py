from turtle import forward
from zmq import device


import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor:
    def __init__(
        self,
        observation_spec,
        action_spec,
        n_hidden_layers,
        conditional_sigma,
        tanh_squash,
    ):
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.n_hidden_layers = n_hidden_layers
        self.conditional_sigma = conditional_sigma
        self.tanh_squash = tanh_squash
        # model layers
        if self.action_spec.is_continuous:

            self.dist = torch.distributions.Normal
        elif self.action_spec.is_discrete:
            self.dist = torch.distributions.Categorical

    def forward(self, state):
        dist = self.dist(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, torch.sum(entropy, dim=1)

    def evaluate(self, state, action):
        dist = self.dist(state)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, torch.sum(entropy, dim=1)


class PocaAgent:
    def __init__(self):
        # value - actor
        self.actor = Actor()
        self.actor.to(device)

        # policy - critic
        self.critic = None

        # optimizer
        self.optimizer = None

        # buffer
        self.buffer = None

    def step(self, state):
        # self.optimizer.reward_signals
        local_rewards = None
        lambda_return = None
        evaluate_result = self.actor.evaluate(state)
        pass

    def train(self):
        batch_size = 32
        adventages = self.buffer.get_advantages(batch_size)
        num_epochs = 10
        for epoch in range(num_epochs):
            # shuffle buffer
            self.buffer = self.buffer.shuffle()
            # train critic
            # update optimizer
