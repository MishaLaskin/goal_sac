import numpy as np

def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()

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
        mean = ptu.from_numpy(self.mean)
        std = ptu.from_numpy(self.std)
        if v.dim() == 2:
            # Unsqueeze along the batch use automatic broadcasting
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return torch.clamp((v - mean) / std, -clip_range, clip_range)

    def denormalize(self, v):
        if not self.synchronized:
            self.synchronize()
        mean = ptu.from_numpy(self.mean)
        std = ptu.from_numpy(self.std)
        if v.dim() == 2:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return mean + v * std


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
