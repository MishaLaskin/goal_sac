import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import utils


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, goal_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
       # self.Q1 = utils.mlp(2*obs_dim-10 + goal_dim + action_dim, hidden_dim, 1, hidden_depth)
       # self.Q2 = utils.mlp(2*obs_dim-10 + goal_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q1 = utils.mlp(64, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(64, hidden_dim, 1, hidden_depth)
        self.outputs = dict()
        self.apply(utils.weight_init)

    def initialize_attention(self, input_module, graph_prop, readout, input_module2, graph_prop2, readout2):
        self.Q1_input_module = input_module
        self.Q1_graph_propagation = graph_prop
        self.Q1_readout = readout
        self.Q2_input_module = input_module2
        self.Q2_graph_propagation = graph_prop2
        self.Q2_readout = readout2

    def forward(self, obs, goal, action):
        assert obs.size(0) == action.size(0)
        obs = torch.cat([obs, goal], dim=-1)
        mask = torch.ones(obs.shape[0], self.goal_dim // 3 - 1).to(self.device)
        Q1_vertices = self.Q1_input_module(obs, actions=action, mask=mask)
        Q1_relational_block_embeddings = self.Q1_graph_propagation.forward(Q1_vertices, mask=mask)
        Q1_pooled_output = self.Q1_readout(Q1_relational_block_embeddings, mask=mask)
       # assert Q1_pooled_output.size(-1) == 1
        Q1_obs = Q1_pooled_output
        Q2_vertices = self.Q2_input_module(obs, actions=action, mask=mask)
        Q2_relational_block_embeddings = self.Q2_graph_propagation.forward(Q2_vertices, mask=mask)
        Q2_pooled_output = self.Q2_readout(Q2_relational_block_embeddings, mask=mask)
       # assert Q2_pooled_output.size(-1) == 1
        Q2_obs = Q2_pooled_output
        #obs = pooled_output.squeeze(1)

       # q1_obs_action = torch.cat([Q1_obs, action], dim=-1)
       # q2_obs_action = torch.cat([Q2_obs, action], dim=-1)
        q1 = self.Q1(Q1_obs)
        q2 = self.Q2(Q2_obs)

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
