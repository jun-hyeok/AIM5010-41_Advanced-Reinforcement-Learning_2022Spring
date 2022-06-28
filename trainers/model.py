from turtle import forward
import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal, Normal
import numpy as np
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class Actor(nn.Module):
    def __init__(self, observation_spec, action_spec):
        super().__init__()
        self.observation_spec = observation_spec  # 20 x 20 x 6 -> [N, 6, 20, 20]
        self.action_spec = action_spec
        input_channels = [space.shape[-1] for space in observation_spec]
        action_dim = action_spec.n
        self.feat1 = nn.Sequential(
            nn.Conv2d(input_channels[0], 24, kernel_size=3),  # type: ignore [N, 6, 20, 20] -> [N, 24, 18, 18]
            nn.ReLU(),
            nn.Conv2d(24, 24, kernel_size=3),  # type: ignore [N, 24, 18, 18] -> [N, 24, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # type: ignore [N, 24, 16, 16] -> [N, 24, 8, 8]
            nn.Linear(24 * 8 * 8, 128),  # type: ignore [N, 24, 8, 8] -> [N, 128]
        )
        self.feat2 = nn.Sequential(
            nn.Linear(input_channels[1], 128),  # type: ignore [N, 10] -> [N, 128]
            nn.ReLU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(128 + 128, 128),  # type: ignore [N, 256] -> [N, 128]
            nn.ReLU(),
            nn.Linear(128, action_dim),  # type: ignore [N, 128] -> [N, action_dim]
        )
        self.sigmma = nn.Sequential(
            nn.Linear(128 + 128, 128),  # type: ignore [N, 256] -> [N, 128]
            nn.ReLU(),
            nn.Linear(128, action_dim),  # type: ignore [N, 128] -> [N, action_dim]
        )

    def forward(self, observation):
        obs1, obs2 = (torch.from_numpy(np.moveaxis(obs,-1,0)).unsqueeze(0) for obs in observation)
        with torch.no_grad():
            x1 = self.feat1(obs1)
            x2 = self.feat2(obs2)
            x = torch.cat((x1, x2), dim=1)
            mu = self.mu(x)
            sigmma = self.sigmma(x)
            sigmma = torch.clamp(sigmma, min=0.01, max=1)
        dist = Normal(mu, sigmma)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action = torch.clamp(action, min=-1, max=1)
        return action.item(), log_prob.item()


class Critic(nn.Module):
    def __init__(self, observation_spec, action_spec):
        super().__init__()
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        input_channels = [space.shape[-1] for space in observation_spec]
        # action_dim = action_spec.shape[-1]
        self.feat1 = nn.Sequential(
            nn.Conv2d(input_channels[0], 24, kernel_size=3),  # type: ignore [N, 6, 20, 20] -> [N, 24, 18, 18]
            nn.ReLU(),
            nn.Conv2d(24, 24, kernel_size=3),  # type: ignore [N, 24, 18, 18] -> [N, 24, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # type: ignore [N, 24, 16, 16] -> [N, 24, 8, 8]
            nn.Linear(24 * 8 * 8, 128),  # type: ignore [N, 24, 8, 8] -> [N, 128]
        )
        self.feat2 = nn.Sequential(
            nn.Linear(input_channels[1], 128),  # type: ignore [N, 10] -> [N, 128]
            nn.ReLU(),
        )
        # self.a = nn.Sequential(
        #     nn.Linear(action_dim, 128),  # type: ignore [N, action_dim] -> [N, 128]
        # )
        self.q = nn.Sequential(
            nn.Linear(128 + 128 + 128, 128),  # type: ignore [N, 256] -> [N, 128]
            nn.ReLU(),
            nn.Linear(128, 1),  # type: ignore [N, 128] -> [N, 1]
        )

    def forward(self, observation):
        observation = (torch.from_numpy(np.moveaxis(obs,-1,0)).unsqueeze(0) for obs in observation)
        obs1, obs2 = observation
        with torch.no_grad():
            x1 = self.feat1(obs1)
            x2 = self.feat2(obs2)
            x = torch.cat((x1, x2), dim=1)
            # a = self.a(action)
            # x = torch.cat((x, a), dim=1)
            q = self.q(x)
        return q


class ActorCritic(nn.Module):
    def __init__(self, observation_spec, action_spec):
        super().__init__()
        self.actor = Actor(observation_spec, action_spec)
        self.critic = Critic(observation_spec, action_spec)

    def forward(self, observation):
        observation = torch.from_numpy(observation)
        _, policy = self.actor.forward(observation)
        value = self.critic(observation)
        return policy, value


# class ActorCritic(nn.Module):
#     def __init__(
#         self, state_dim, action_dim, has_continuous_action_space, action_std_init
#     ):
#         super(ActorCritic, self).__init__()

#         self.has_continuous_action_space = has_continuous_action_space

#         if has_continuous_action_space:
#             self.action_dim = action_dim
#             self.action_var = torch.full(
#                 (action_dim,), action_std_init * action_std_init
#             ).to(device)
#         # actor
#         if has_continuous_action_space:
#             self.actor = nn.Sequential(
#                 nn.Linear(state_dim, 64),
#                 nn.Tanh(),
#                 nn.Linear(64, 64),
#                 nn.Tanh(),
#                 nn.Linear(64, action_dim),
#             )
#         else:
#             self.actor = nn.Sequential(
#                 nn.Linear(state_dim, 64),
#                 nn.Tanh(),
#                 nn.Linear(64, 64),
#                 nn.Tanh(),
#                 nn.Linear(64, action_dim),
#                 nn.Softmax(dim=-1),
#             )
#         # critic
#         self.critic = nn.Sequential(
#             nn.Linear(state_dim, 64),
#             nn.Tanh(),
#             nn.Linear(64, 64),
#             nn.Tanh(),
#             nn.Linear(64, 1),
#         )

#     def set_action_std(self, new_action_std):
#         if self.has_continuous_action_space:
#             self.action_var = torch.full(
#                 (self.action_dim,), new_action_std * new_action_std
#             ).to(device)
#         else:
#             print(
#                 "--------------------------------------------------------------------------------------------"
#             )
#             print(
#                 "WARNING : Calling ActorCritic::set_action_std() on discrete action space policy"
#             )
#             print(
#                 "--------------------------------------------------------------------------------------------"
#             )

#     def forward(self):
#         raise NotImplementedError

#     def act(self, state):
#         if self.has_continuous_action_space:
#             action_mean = self.actor(state)
#             cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
#             dist = MultivariateNormal(action_mean, cov_mat)
#         else:
#             action_probs = self.actor(state)
#             dist = Categorical(action_probs)

#         action = dist.sample()
#         action_logprob = dist.log_prob(action)

#         return action.detach(), action_logprob.detach()

#     def evaluate(self, state, action):

#         if self.has_continuous_action_space:
#             action_mean = self.actor(state)

#             action_var = self.action_var.expand_as(action_mean)
#             cov_mat = torch.diag_embed(action_var).to(device)
#             dist = MultivariateNormal(action_mean, cov_mat)

#             # For Single Action Environments.
#             if self.action_dim == 1:
#                 action = action.reshape(-1, self.action_dim)
#         else:
#             action_probs = self.actor(state)
#             dist = Categorical(action_probs)
#         action_logprobs = dist.log_prob(action)
#         dist_entropy = dist.entropy()
#         state_values = self.critic(state)

#         return action_logprobs, state_values, dist_entropy
