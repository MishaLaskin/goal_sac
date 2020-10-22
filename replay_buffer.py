import numpy as np
import torch

def random_crop(imgs, out=78):
    """
        args:
        imgs: np.array shape (B,C,H,W)
        out: output size (e.g. 84)
        returns np.array
    """
    n, c, h, w = imgs.shape
    crop_max = h - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = np.empty((n, c, out, out), dtype=np.float32)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):

        cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
    return cropped

def center_crop_image(image, output_size=78):
    h, w = image.shape[1:]
    if h > output_size: #center cropping
        new_h, new_w = output_size, output_size

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        image = image[:, top:top + new_h, left:left + new_w]
        return image
    else: #center translate
        new_image = np.zeros((image.shape[0], output_size, output_size))
        shift = output_size - h
        shift = shift // 2
        new_image[:, shift:shift + h, shift:shift+w] = image
        return new_image

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, goal_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 #if len(obs_shape) == 1 else np.uint8

        self.obs_shape = obs_shape 
        self.goal_shape = goal_shape

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)

        self.achieved_goals = np.empty((capacity, *goal_shape), dtype=obs_dtype)
        self.desired_goals = np.empty((capacity, *goal_shape), dtype=obs_dtype)

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
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs['observation'])
        np.copyto(self.achieved_goals[self.idx], next_obs['achieved_goal'])
        np.copyto(self.desired_goals[self.idx], next_obs['desired_goal'])
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_crop(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)
        obses = self.obses[idxs]
        obses = random_crop(obses)
        next_obses = self.next_obses[idxs]
        next_obses = random_crop(next_obses)
        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(next_obses,
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        achieved_goals = torch.as_tensor(self.achieved_goals[idxs], device=self.device).float()
        desired_goals = torch.as_tensor(self.desired_goals[idxs], device=self.device).float()

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max, achieved_goals, desired_goals

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