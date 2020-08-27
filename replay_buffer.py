import numpy as np
import torch


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, goal_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obs_shape = obs_shape 
        self.goal_shape = goal_shape

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)

        self.achieved_goals = np.empty((capacity, *goal_shape), dtype=obs_dtype)
        self.desired_goals = np.empty((capacity, *goal_shape), dtype=obs_dtype)

        print("ACTION SHAPE", action_shape)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs['observation'])
        #np.copyto(self.actions[self.idx], action)
        self.actions[self.idx] = action
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs['observation'])
        np.copyto(self.achieved_goals[self.idx], next_obs['achieved_goal'])
        np.copyto(self.desired_goals[self.idx], next_obs['desired_goal'])
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        achieved_goals = torch.as_tensor(self.achieved_goals[idxs], device=self.device).float()
        desired_goals = torch.as_tensor(self.desired_goals[idxs], device=self.device).float()

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max, achieved_goals, desired_goals