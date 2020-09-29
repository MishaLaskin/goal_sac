import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)
import torch
import numpy as np
from agent.normalizer import CompositeNormalizer

def create_attention_embedding(device, qval=False, shared_normalizer=None):
    action_dim, object_dim, goal_dim, shared_dim = 4, 15, 3, 10
    dim = object_dim + shared_dim + goal_dim
    embedding_dim=64
    ret_norm = False
    if not shared_normalizer:
        ret_norm = True
        shared_normalizer = CompositeNormalizer(object_dim + shared_dim + goal_dim,
                                                action_dim,
                                                default_clip_range=5,
                                                reshape_blocks=True,
                                                fetch_kwargs=dict(
                                                    lop_state_dim=3,
                                                    object_dim=object_dim,
                                                    goal_dim=goal_dim
                                                ))
    if qval:
        dim += action_dim
    input_module_kwargs = dict(
        normalizer=shared_normalizer,
        object_total_dim=dim,
        embedding_dim=64,
        layer_norm=True
    )
    graphprop_kwargs =  dict(
        graph_module_kwargs=dict(
            embedding_dim=64,
            num_heads=1,
        ),
        layer_norm=True,
        num_query_heads=1,
        num_relational_blocks=3,
        activation_fnx=F.leaky_relu,
        recurrent_graph=False
    )
    mlp_kwargs = None
    if qval:
        mlp_kwargs = dict(
        hidden_sizes=[64, 64, 64],
        output_size=1,
        input_size=1*embedding_dim,
        layer_norm=True,
        )
    input_module = FetchInputPreprocessing(**input_module_kwargs, device=device)
    graph_propagation = GraphPropagation(**graphprop_kwargs)
    readout = AttentiveGraphPooling(**mlp_kwargs)
    if ret_norm:
        return input_module, graph_propagation, readout, shared_normalizer
    return input_module, graph_propagation, readout
    # vertices = self.input_module(obs, actions=actions, mask=mask)
    # relational_block_embeddings = self.graph_propagation.forward(vertices, mask=mask)
    # pooled_output = self.readout(relational_block_embeddings, mask=mask)

class FetchInputPreprocessing(nn.Module):
    """
    Used for the Q-value and value function
    Takes in either obs or (obs, actions) in the forward function and returns the same sized embedding for both
    Make sure actions are being passed in!!
    """
    def __init__(self,
                 normalizer,
                 object_total_dim,
                 embedding_dim,
                 device,
                 layer_norm=True):
        self.save_init_params(locals())
        super().__init__()
        self.normalizer = normalizer
        self.fc_embed = nn.Linear(object_total_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim) if layer_norm else None
        self.device = device

    def forward(self, obs, actions=None, mask=None):
        vertices = fetch_preprocessing(obs, self.device, actions=actions, normalizer=self.normalizer, mask=mask)

        if self.layer_norm is not None:
            return self.layer_norm(self.fc_embed(vertices))
        else:
            return self.fc_embed(vertices)


def fetch_preprocessing(obs,
                        device,
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
        obs = torch.zeros(batch_size, environment_state_length).to(device)

    nB = (environment_state_length - robot_dim) / (object_dim + goal_dim)

    assert nB.is_integer(), (nB, environment_state_length, robot_dim, object_dim, goal_dim) # TODO: this checks if the lopped state still breaks down into the right object dimensions. The only worry here is whether the obs was messed up at the start of the function, e.g. the samples from the replay buffer incorrectly put the lopped state somewwhere.

    nB = int(nB)
    if mask is None:
        mask = torch.ones(obs.shape[0], nB).to(device)

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
    batch_objgoals = torch.cat((batched_objects, batched_goals), dim=-1)

    batch_shared = batch_shared.clone() * mask.unsqueeze(-1).expand_as(batch_shared)
    batch_objgoals = batch_objgoals.clone() * mask.unsqueeze(-1).expand_as(batch_objgoals)

    if normalizer is not None:
        robot_singleobj_singlegoal = torch.cat((batch_shared, batch_objgoals), dim=-1).view(batch_size * nB, robot_dim + object_dim + goal_dim)

        # Single objects means, we flatten the nB dimension
        norm_singlerobot_singleobj_singlegoal, norm_actions = normalizer.normalize_all(robot_singleobj_singlegoal, actions)

        # Set these two variables to be the normalized versions
        norm_singlerobot, norm_singleobj_singlegoal = torch.split(norm_singlerobot_singleobj_singlegoal, [robot_dim, object_dim + goal_dim], dim=-1)

        # Turn single objects back into batches of nB objects
        norm_batchobjgoals = norm_singleobj_singlegoal.contiguous().view(batch_size, nB,  object_dim + goal_dim)
        norm_batchshared = norm_singlerobot.contiguous().view(batch_size, nB, robot_dim)

        batch_shared = norm_batchshared
        batch_objgoals = norm_batchobjgoals
        actions = norm_actions

    if actions is not None:
        batch_shared = torch.cat((actions.unsqueeze(1).expand(-1, nB, -1), batch_shared), dim=-1)

    assert batch_shared.shape == torch.Size([batch_size, nB, robot_dim + action_dim]), (batch_shared.shape, torch.Size([batch_size, nB, robot_dim + action_dim]))

    batched_combined_state = torch.cat((batch_shared, batch_objgoals), dim=-1)
    return batched_combined_state



class Attention(nn.Module):
    """
    Additive, multi-headed attention
    """
    def __init__(self,
                 embedding_dim,
                 num_heads=1,
                 layer_norm=True,
                 activation_fnx=F.leaky_relu,
                 softmax_temperature=1.0):
        self.save_init_params(locals())
        super().__init__()
        self.fc_createheads = nn.Linear(embedding_dim, num_heads * embedding_dim)
        self.fc_logit = nn.Linear(embedding_dim, 1)
        self.fc_reduceheads = nn.Linear(num_heads * embedding_dim, embedding_dim)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(i) for i in [num_heads*embedding_dim, 1, embedding_dim]]) if layer_norm else None
        self.softmax_temperature = nn.Parameter(torch.tensor(softmax_temperature))

        self.activation_fnx = activation_fnx

    def forward(self, query, context, memory, mask):
        """
        N, nV, nE memory -> N, nV, nE updated memory
        :param query:
        :param context:
        :param memory:
        :param mask: N, nV
        :return:
        """
        N, nQ, nE = query.size()
        # assert len(query.size()) == 3

        # assert self.fc_createheads.out_features % nE == 0
        nH = int(self.fc_createheads.out_features / nE)

        nV = memory.size(1)

        # assert len(mask.size()) == 2

        # N, nQ, nE -> N, nQ, nH, nE
        # if nH > 1:
        query = self.fc_createheads(query).view(N, nQ, nH, nE)

        # N, nQ, nH, nE -> N, nQ, nV, nH, nE
        query = query.unsqueeze(2).expand(-1, -1, nV, -1, -1)

        # N, nV, nE -> N, nQ, nV, nH, nE
        context = context.unsqueeze(1).unsqueeze(3).expand_as(query)

        # -> N, nQ, nV, nH, 1
        qc_logits = self.fc_logit(torch.tanh(context + query))

        # if self.layer_norms is not None:
        #     qc_logits = self.layer_norms[1](qc_logits)

        # N, nV -> N, nQ, nV, nH, 1
        logit_mask = mask.unsqueeze(1).unsqueeze(3).unsqueeze(-1).expand_as(qc_logits)

        # qc_logits N, nQ, nV, nH, 1 -> N, nQ, nV, nH, 1
        attention_probs = F.softmax(qc_logits / self.softmax_temperature * logit_mask + (-99999) * (1 - logit_mask), dim=2)

        # N, nV, nE -> N, nQ, nV, nH, nE
        memory = memory.unsqueeze(1).unsqueeze(3).expand(-1, nQ, -1, nH, -1)

        # N, nV -> N, nQ, nV, nH, nE
        memory_mask = mask.unsqueeze(1).unsqueeze(3).unsqueeze(4).expand_as(memory)

        # assert memory.size() == attention_probs.size() == mask.size(), (memory.size(), attention_probs.size(), memory_mask.size())

        # N, nQ, nV, nH, nE -> N, nQ, nH, nE
        attention_heads = (memory * attention_probs * memory_mask).sum(2).squeeze(2)

        attention_heads = self.activation_fnx(attention_heads)
        # N, nQ, nH, nE -> N, nQ, nE
        attention_result = self.fc_reduceheads(attention_heads.view(N, nQ, nH*nE))

        return attention_result


class AttentiveGraphToGraph(nn.Module):
    """
    Uses attention to perform message passing between 1-hop neighbors in a fully-connected graph
    """
    def __init__(self,
                 embedding_dim=64,
                 num_heads=1,
                 layer_norm=True,
                 **kwargs):
        self.save_init_params(locals())
        super().__init__()
        self.fc_qcm = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.attention = Attention(embedding_dim, num_heads=num_heads, layer_norm=layer_norm)
        self.layer_norm= nn.LayerNorm(3*embedding_dim) if layer_norm else None

    def forward(self, vertices, mask):
        """
        :param vertices: N x nV x nE
        :return: updated vertices: N x nV x nE
        """
        assert len(vertices.size()) == 3
        N, nV, nE = vertices.size()
        assert mask.size() == torch.Size([N, nV])

        # -> (N, nQ, nE), (N, nV, nE), (N, nV, nE)

        qcm_block = self.fc_qcm(vertices)

        query, context, memory = qcm_block.chunk(3, dim=-1)

        return self.attention(query, context, memory, mask)


class AttentiveGraphPooling(nn.Module):
    """
    Pools nV vertices to a single vertex embedding
    """
    def __init__(self,
                 embedding_dim=64,
                 num_heads=1,
                 init_w=3e-3,
                 layer_norm=True,
                 mlp_kwargs=None):
        self.save_init_params(locals())
        super().__init__()
        self.fc_cm = nn.Linear(embedding_dim, 2 * embedding_dim)
        self.layer_norm = nn.LayerNorm(2*embedding_dim) if layer_norm else None

        self.input_independent_query = nn.Parameter(torch.Tensor(embedding_dim))
        self.input_independent_query.data.uniform_(-init_w, init_w)
        # self.num_heads = num_heads
        self.attention = Attention(embedding_dim, num_heads=num_heads, layer_norm=layer_norm)

        if mlp_kwargs is not None:
            self.proj = Mlp(**mlp_kwargs)
        else:
            self.proj = None

    def forward(self, vertices, mask):
        """
        N, nV, nE -> N, nE
        :param vertices:
        :param mask:
        :return:
        """
        N, nV, nE = vertices.size()

        # nE -> N, nQ, nE where nQ == self.num_heads
        query = self.input_independent_query.unsqueeze(0).unsqueeze(0).expand(N, 1, -1)

        context = vertices
        memory = vertices

        attention_result = self.attention(query, context, memory, mask)

        if self.proj is not None:
            return self.proj(attention_result).squeeze(1)
        else:
            return attention_result

def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

class Mlp(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=lambda x: x,
            hidden_init=fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        if isinstance(output_activation, str):
            output_activation = getattr(torch, output_activation)
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = nn.ModuleList([])
        self.layer_norms = nn.ModuleList([])
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = nn.LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
            if self.layer_norm and i < len(self.fcs):
                h = self.layer_norms[i](h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

class GraphPropagation(nn.Module):
    """
    Input: state
    Output: context vector
    """

    def __init__(self,
                 num_relational_blocks=1,
                 num_query_heads=1,
                 graph_module_kwargs=None,
                 layer_norm=False,
                 activation_fnx=F.leaky_relu,
                 graph_module=AttentiveGraphToGraph,
                 post_residual_activation=True,
                 recurrent_graph=False,
                 **kwargs
                 ):
        """
        :param embedding_dim:
        :param lstm_cell_class:
        :param lstm_num_layers:
        :param graph_module_kwargs:
        :param style: OSIL or relational inductive bias.
        """
        self.save_init_params(locals())
        super().__init__()

        # Instance settings

        self.num_query_heads = num_query_heads
        self.num_relational_blocks = num_relational_blocks
        assert graph_module_kwargs, graph_module_kwargs
        self.embedding_dim = graph_module_kwargs['embedding_dim']

        if recurrent_graph:
            rg = graph_module(**graph_module_kwargs)
            self.graph_module_list = nn.ModuleList(
                [rg for i in range(num_relational_blocks)])
        else:
            self.graph_module_list = nn.ModuleList(
                [graph_module(**graph_module_kwargs) for i in range(num_relational_blocks)])

        # Layer norm takes in N x nB x nE and normalizes
        if layer_norm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(self.embedding_dim) for i in range(num_relational_blocks)])

        # What's key here is we never use the num_objects in the init,
        # which means we can change it as we like for later.

        """
        ReNN Arguments
        """
        self.layer_norm = layer_norm
        self.activation_fnx = activation_fnx

    def forward(self, vertices, mask=None, *kwargs):
        """
        :param shared_state: state that should be broadcasted along nB dimension. N * (nR + nB * nF)
        :param object_and_goal_state: individual objects
        :return:
        """
        output = vertices

        for i in range(self.num_relational_blocks):
            new_output = self.graph_module_list[i](output, mask)
            new_output = output + new_output

            output = self.activation_fnx(new_output) # Diff from 7/22
            # Apply layer normalization
            if self.layer_norm:
                output = self.layer_norms[i](output)
        return output
