import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import utils


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, goal_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.Q1 = utils.mlp(2*obs_dim + goal_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(2*obs_dim + goal_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, goal, action):
        assert obs.size(0) == action.size(0)
        block_size = 15
        block_pos = torch.narrow(obs, 1, 10, self.obs_dim - 10)
        block_pos = block_pos.view(block_pos.shape[0], block_pos.shape[-1] // block_size, block_size)
        attention_block = self.attention_blocks(block_pos)
        obs = torch.cat([obs, attention_block, goal], dim=-1)

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
