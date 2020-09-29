import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)
import torch
import rlkit.torch.pytorch_util as ptu
import numpy as np


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, block_size=1, **kwargs):
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class BlockConfig(GPTConfig):
    n_layer=4
    n_head=3
    n_embd=15 # double check this

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

def fetch_preprocessing(obs,
                        actions=None,
                        normalizer=None,
                        robot_dim=10,
                        object_dim=15,
                        goal_dim=3,
                        zero_state_preprocessing_fnx=False,
                        lop_state_dim=3,
                        mask=None,
                        return_combined_state=True):
    """
    For Fetch robotics gym environment. Takes a flattened state and processes it into batched, normalized objects.
    :param obs: N x (nR + nB * nFb)
    :param actions
    :param robot_dim:
    :param object_dim: nFb
    :param num_objects:
    :param goal_dim:
    :param zero_state_preprocessing_fnx: Zero out state for testing.
    :return: N x nB x (nFb + nR). If in QValueCPA, concats actions to the left of the shared return
    """
    if len(obs.shape) == 3:
        obs = obs.squeeze(1)
    if lop_state_dim:
        obs = obs.narrow(1, 0, obs.size(1)-lop_state_dim) # Chop off the final 3 dimension of gripper position

    batch_size, environment_state_length = obs.size()
    if actions is not None:
        action_dim = actions.size(-1)
    else:
        action_dim = 0

    if zero_state_preprocessing_fnx:
        obs = torch.zeros(batch_size, environment_state_length).to(ptu.device)

    nB = (environment_state_length - robot_dim) / (object_dim + goal_dim)

    assert nB.is_integer(), (nB, environment_state_length, robot_dim, object_dim, goal_dim) # TODO: this checks if the lopped state still breaks down into the right object dimensions. The only worry here is whether the obs was messed up at the start of the function, e.g. the samples from the replay buffer incorrectly put the lopped state somewwhere.

    nB = int(nB)
    if mask is None:
        mask = torch.ones(obs.shape[0], nB).to(ptu.get_device())

    kwargs_state_length = robot_dim + object_dim * nB + goal_dim * nB
    assert kwargs_state_length == environment_state_length, F"{kwargs_state_length} != {environment_state_length}"

    # N x nR. From index 0 to shared dim per sample, we have the robot_state
    robot_state_flat = obs.narrow(1, 0, robot_dim)

    # assert (state_length - shared_dim - goal_state_dim) % block_feature_dim == 0, state_length - shared_dim - goal_state_dim

    # N x (nB x nFb)
    flattened_objects = obs.narrow(1, robot_dim, object_dim * nB)

    # -> N x nB x nFb
    batched_objects = flattened_objects.view(batch_size, nB, object_dim)

    # N x (nB x nFg) # TODO: perhaps add lop state dim
    flattened_goals = obs.narrow(1, robot_dim + nB * object_dim, nB * goal_dim)

    # -> N x nB x nFg
    batched_goals = flattened_goals.view(batch_size, nB, goal_dim)

    assert torch.eq(torch.cat((
                         robot_state_flat.view(batch_size, -1),
                         batched_objects.view(batch_size, -1),
                         batched_goals.view(batch_size, -1)), dim=1),
        obs).all()

    # Broadcast robot_state
    # -> N x nB x nR
    batch_shared = robot_state_flat.unsqueeze(1).expand(-1, nB, -1)

    # Concatenate with block_state
    # N x nB x (nFb + nR)
    # output_state = torch.cat((block_state, robot_state), dim=2)
    # return output_state

    # We can just consider the goals to be part of the block state, so we concat them together

    batch_objgoals = torch.cat((batched_objects, batched_goals), dim=-1)

    batch_shared = batch_shared.clone() * mask.unsqueeze(-1).expand_as(batch_shared)
    batch_objgoals = batch_objgoals.clone() * mask.unsqueeze(-1).expand_as(batch_objgoals)
    # assert torch.unique(batch_shared, dim=1).shape == torch.Size([batch_size, 1, robot_dim]), (
    # torch.unique(batch_shared, dim=1).shape, torch.Size([batch_size, 1, robot_dim]))

    if normalizer is not None:
        robot_singleobj_singlegoal = torch.cat((batch_shared, batch_objgoals), dim=-1).view(batch_size * nB, robot_dim + object_dim + goal_dim)

        # Single objects means, we flatten the nB dimension
        norm_singlerobot_singleobj_singlegoal, norm_actions = normalizer.normalize_all(robot_singleobj_singlegoal, actions)

        # Set these two variables to be the normalized versions
        norm_singlerobot, norm_singleobj_singlegoal = torch.split(norm_singlerobot_singleobj_singlegoal, [robot_dim, object_dim + goal_dim], dim=-1)

        # Turn single objects back into batches of nB objects
        norm_batchobjgoals = norm_singleobj_singlegoal.contiguous().view(batch_size, nB,  object_dim + goal_dim)
        norm_batchshared = norm_singlerobot.contiguous().view(batch_size, nB, robot_dim)
        # assert torch.unique(norm_batchshared, dim=1).shape == torch.Size([batch_size, 1, robot_dim]), (torch.unique(norm_batchshared, dim=1).shape, torch.Size([batch_size, 1, robot_dim]))

        batch_shared = norm_batchshared
        batch_objgoals = norm_batchobjgoals
        actions = norm_actions

    if actions is not None:
        batch_shared = torch.cat((actions.unsqueeze(1).expand(-1, nB, -1), batch_shared), dim=-1)

    assert batch_shared.shape == torch.Size([batch_size, nB, robot_dim + action_dim]), (batch_shared.shape, torch.Size([batch_size, nB, robot_dim + action_dim]))

    if return_combined_state:
        batched_combined_state = torch.cat((batch_shared, batch_objgoals), dim=-1)
        return batched_combined_state
    else:
        return batch_shared, batch_objgoals