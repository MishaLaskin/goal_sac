from collections import namedtuple

import numpy as np
from gym import spaces
import gym
from skimage.transform import resize

import os
import copy
from gym.utils import seeding
import matplotlib.pyplot as plt
from random import shuffle


ACTION_MEANING = {
    0: "RIGHT",
    1: "LEFT",
    2: "UP",
    3: "DOWN",
    4: "RIGHTUP",
    5: "RIGHTDOWN",
    6: "LEFTUP",
    7: "LEFTDOWN",
}

def make_block_env():
    env = BlockPlaceEnv()
    env = ActionToNum(env)
    env = DictObsWrap(env)
    return env

class DictObsWrap(gym.ObservationWrapper):
    def __init__(self, env, block=1):
        self.env = env
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=0, high=env.num_blocks, shape=(1, *env.obs_shape),  dtype=np.uint8),
            'desired_goal': spaces.Box(low=0, high=env.num_blocks, shape=(1, *env.obs_shape),  dtype=np.uint8),
            'achieved_goal': spaces.Box(low=0, high=1, shape=(1, *env.obs_shape),  dtype=np.uint8)
        })
        self.action_space = env.action_space
        self.goal_pos = env.env.goal_pos
        self._max_episode_steps = env.env._max_episode_steps
        self.block = block

    def observation(self, obs):
        desired_goal = self.env.env.grid.copy()
        block_pos = self.env.env._get_block_pos(self.block)
        desired_goal[block_pos[0]][block_pos[1]] = 0
        goal_pos = np.array(self.env.env.goal_pos).astype(np.uint8)
        desired_goal[goal_pos[0]][goal_pos[1]] = self.block
        return {'observation': np.expand_dims(obs, 0),
                'desired_goal': np.expand_dims(desired_goal, 0),
                'achieved_goal': np.expand_dims(obs, 0)}


class BlockEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.seed()

        self._action_set = [0, 1, 2, 3, 4, 5, 6, 7]
        self.action_space = spaces.Discrete(len(self._action_set))
        self.act_to_ij = {0: [0, 1], 1: [0, -1], 2: [-1, 0], 3: [1, 0], 4: [-1, 1], 5: [1, 1], 6: [-1, -1], 7: [1, -1]}
        # set observation space
        self.num_blocks = 3
        self.blocks = list(range(1, self.num_blocks + 1))
        self.grid_size = 10
        self.box_size = 1
        self.obs_shape = [self.grid_size, self.grid_size]
        self.observation_space = spaces.Box(low=0, high=self.num_blocks, shape=self.obs_shape, dtype=np.uint8)
        self._max_episode_steps = 25 # self.num_blocks * 25
        self.use_goal = True
        self.reset()

    def reset(self):
        self.reset_goal()
        self._step = 0
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        n = self.num_blocks
        objects = list(range(1, n + 1))
        for m in objects:
            while True:
                i, j = np.random.randint(grid.shape[0], size=2)
                if grid[i, j] == 0 and not (np.array([i, j]) == self.goal_pos).all():
                    grid[i, j] = m
                    break
        self.grid = grid
        obs = grid.copy()
        return obs

    def step(self, action_block):
        action, block = action_block
        info = {}
        info['success'] = False
        self._step += 1
        i, j = np.argwhere(self.grid == block)[0]
        delta = self.act_to_ij[action]
        new_ij = [(i + delta[0]) % (self.grid_size), (j + delta[1]) % (self.grid_size)]
        if not self._is_collision(new_ij[0], new_ij[1]):
            self._move_block(new_ij, block)

        next_obs = self.grid.copy()
        success = self._is_done(next_obs)
        done = success or self._step >= self._max_episode_steps
        reward = 0 if success else -1
        if done:
            pass
            #print("done", done, reward)

        return next_obs, reward, done, info

    def _is_collision(self, i, j):
        return self.grid[i, j] != 0

    def _get_block_pos(self, block):
        return np.array(np.argwhere(self.grid == block)[0])

    def _get_block_goal_pos(self, block):
        return np.array(np.argwhere(self.goal == block)[0])

    def _move_block(self, new_ij, block):
        if not self._is_collision(*new_ij):
            prev_ij = self._get_block_pos(block)
            self.grid[prev_ij[0], prev_ij[1]] = 0
            self.grid[new_ij[0], new_ij[1]] = block
        return self.grid.copy()

    def _is_done(self, obs):
        raise NotImplementedError
        # checks if we have a stack
        ids = np.argwhere(obs > 0)
        consec = ids[:, np.argwhere(ids.mean(0) - ids.min(0) > 0)[0][0]]
        diff = consec - np.arange(consec[0], consec[0] + len(consec))
        return True if not diff.any() else False

    def reset_goal(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        order = list(range(1, self.num_blocks + 1))
        shuffle(order)

        vertical = np.random.randint(2)
        start = np.random.randint(self.grid_size - self.num_blocks)
        i_s = [start + m for m in range(self.num_blocks)]
        j = np.random.randint(self.grid_size)
        if vertical:
            ijs = [[i, j] for i in i_s]
        else:
            ijs = [[j, i] for i in i_s]

        for index, ij in enumerate(ijs):
            i, j = ij[0], ij[1]
            grid[i, j] = order[index]
        self.goal = grid

    def render(self, mode=None):
        if mode == 'rgb_array':
            #img = resize(self.grid, (60,60)).astype(np.uint8)
            img = self.grid
            img = img * (255//4)
            img = np.stack((img, img, img)).transpose(1, 2, 0)
        else:
            img = self.grid
        return img

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

    def compute_reward(self, achieved_goal, desired_goal, info):
        if (achieved_goal == desired_goal).all():
            return 0
        return -1


class ActionToNum(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_actions = int(env.action_space.n)
        self.action_space = spaces.Discrete(env.action_space.n ) #

    def action(self, ac):
        new_ac = ac % self.num_actions
        block = (ac // self.num_actions) + 1
       # print(new_ac, block)
        return new_ac, block

class BlockPlaceEnv(BlockEnv):
    def reset_goal(self):
        self.goal_pos = ((np.random.randint(self.grid_size), np.random.randint(self.grid_size)))
        self.block_ind = 1 #np.random.randint(self.num_blocks)

    def _is_done(self, obs):
        # print(self._get_block_pos(self.block_ind))
        # print(self.goal_pos)
        # print((self._get_block_pos(self.block_ind) == self.goal_pos).all())
        return (self._get_block_pos(self.block_ind) == self.goal_pos).all()