import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from agent.encoder import make_encoder

import utils


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, goal_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()
        feature_dim = 50
        self.encoder = make_encoder(
            'pixel', obs_dim, feature_dim=50, num_layers=4,
            num_filters=3, output_logits=True, two_conv=False
        )

        self.Q1 = utils.mlp(feature_dim + goal_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(feature_dim + goal_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, goal, action, detach_encoder=False):
        assert obs.size(0) == action.size(0)
        obs = self.encoder(obs, detach=detach_encoder)
     
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
