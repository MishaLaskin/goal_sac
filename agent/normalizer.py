import numpy as np

import torch

def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
def from_numpy(*args, device=None, **kwargs):
    if device is None:
        return torch.from_numpy(*args, **kwargs).float().to(torch.device('cuda'))
    else:
        return torch.from_numpy(*args, **kwargs).float().to(device)

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
    return batched_combined_state.to(device)

class Normalizer(object):
    def __init__(
            self,
            size,
            eps=1e-8,
            default_clip_range=np.inf,
            mean=0,
            std=1,
    ):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        self.sum = np.zeros(self.size, np.float32)
        self.sumsq = np.zeros(self.size, np.float32)
        self.count = np.ones(1, np.float32)
        self.mean = mean + np.zeros(self.size, np.float32)
        self.std = std * np.ones(self.size, np.float32)
        self.synchronized = True

    def update(self, v):
        if v.ndim == 1:
            v = np.expand_dims(v, 0)
        assert v.ndim == 2
        assert v.shape[1] == self.size
        self.sum += v.sum(axis=0)
        self.sumsq += (np.square(v)).sum(axis=0)
        self.count[0] += v.shape[0]
        self.synchronized = False

    def normalize(self, v, clip_range=None):
        if not self.synchronized:
            self.synchronize()
        if clip_range is None:
            clip_range = self.default_clip_range
        mean, std = self.mean, self.std
        if v.ndim == 2:
            mean = mean.reshape(1, -1)
            std = std.reshape(1, -1)
        return np.clip((v - mean) / std, -clip_range, clip_range)

    def denormalize(self, v):
        if not self.synchronized:
            self.synchronize()
        mean, std = self.mean, self.std
        if v.ndim == 2:
            mean = mean.reshape(1, -1)
            std = std.reshape(1, -1)
        return mean + v * std

    def synchronize(self):
        self.mean[...] = self.sum / self.count[0]
        self.std[...] = np.sqrt(
            np.maximum(
                np.square(self.eps),
                self.sumsq / self.count[0] - np.square(self.mean)
            )
        )
        self.synchronized = True

class CompositeNormalizer:
    """
    Useful for normalizing different data types e.g. when using the same normalizer for the Q function and the policy function
    """

    def __init__(self,
                 obs_dim,
                 action_dim,
                 reshape_blocks=False,
                 fetch_kwargs=dict(),
                 **kwargs):
        # self.save_init_params(locals())
        self.observation_dim = obs_dim
        self.action_dim = action_dim
        self.obs_normalizer = TorchNormalizer(self.observation_dim, **kwargs)
        self.action_normalizer = TorchNormalizer(self.action_dim)
        self.reshape_blocks = reshape_blocks
        self.kwargs = kwargs
        self.fetch_kwargs = fetch_kwargs

    def normalize_all(
            self,
            flat_obs,
            actions):
        """
        :param flat_obs:
        :param actions:
        :return:
        """
        if flat_obs is not None:
            flat_obs = self.obs_normalizer.normalize(flat_obs)
        if actions is not None:
            actions = self.action_normalizer.normalize(actions)
        return flat_obs, actions

    def update(self, data_type, v, mask=None):
        """
        Takes in tensor and updates numpy array
        :param data_type:
        :param v:
        :return:
        """
        if data_type == "obs":
            # Reshape_blocks: takes flat, turns batch, normalizes batch, updates the obs_normalizer...
            if self.reshape_blocks:
                batched_robot_state, batched_objects_and_goals = fetch_preprocessing(v, mask=mask,
                                                                                     return_combined_state=False,
                                                                                     **self.fetch_kwargs)
                N, nB, nR = batched_robot_state.size()
                v = torch.cat((batched_robot_state, batched_objects_and_goals), dim=-1).view(N * nB, -1)
                if mask is not None:
                    v = v[mask.view(N * nB).to(dtype=torch.bool)]

            # if self.lop_state_dim:
            #     v = v.narrow(-1, -3, 3)
            self.obs_normalizer.update(get_numpy(v))
        elif data_type == "actions":
            self.action_normalizer.update(get_numpy(v))
        else:
            raise ("data_type not set")


class TorchNormalizer(Normalizer):
    """
    Update with np array, but de/normalize pytorch Tensors.
    """

    def normalize(self, v, clip_range=None):
        if not self.synchronized:
            self.synchronize()
        if clip_range is None:
            clip_range = self.default_clip_range
        mean = from_numpy(self.mean)
        std = from_numpy(self.std)
        if v.dim() == 2:
            # Unsqueeze along the batch use automatic broadcasting
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return torch.clamp((v - mean) / std, -clip_range, clip_range)

    def denormalize(self, v):
        if not self.synchronized:
            self.synchronize()
        mean = from_numpy(self.mean)
        std = from_numpy(self.std)
        if v.dim() == 2:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return mean + v * std


