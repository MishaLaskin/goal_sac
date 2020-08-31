import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from agent.encoder import PixelEncoder

import utils


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, goal_dim, action_dim, hidden_dim, hidden_depth, encoder=None):
        super().__init__()
        self.encoder = encoder

        self.Q1 = utils.mlp(obs_dim + goal_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(obs_dim + goal_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, goal, action):
        assert obs.size(0) == action.size(0)
        obs = self.encoder(obs.float())
        obs = torch.cat([obs, goal], dim=-1)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)

class DiscreteCritic(nn.Module):
    def __init__(self, obs_dim, goal_dim, action_dim, hidden_dim, hidden_depth,  action_range):
        super().__init__()
        self.encoder = PixelEncoder((3, 10, 10), 128)

        encoder_out = 128 # todo: this is currently hard coded
        self.Q1 = utils.mlp(encoder_out * 2, hidden_dim, action_range, hidden_depth)
        self.Q2 = utils.mlp(encoder_out * 2 , hidden_dim, action_range, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, goal, detach_encoder=False):
        obs = self.encoder(obs.float(), detach=detach_encoder)
        goal = self.encoder(goal.float(), detach=detach_encoder)
        obs = torch.cat([obs, goal], dim=-1)

        q1 = self.Q1(obs)
        q2 = self.Q2(obs)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)
