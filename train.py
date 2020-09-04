#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl

#import fetch_block_construction


from video import VideoRecorder, GridVideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils
from envs import block_env

import dmc2gym
import hydra



class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        #self.env = utils.make_env(cfg)
        self.env = block_env.make_block_env()
        self.obs_shape = self.env.observation_space['observation'].shape
        self.goal_shape = self.env.observation_space['desired_goal'].shape

        cfg.agent.params.obs_dim = self.obs_shape[0]
        print("OBS DIM", cfg.agent.params.obs_dim)
        cfg.agent.params.goal_dim = self.goal_shape[0]
        cfg.agent.params.action_dim = 1 #self.env.action_space.#shape[0]
        cfg.agent.params.action_range = self.env.action_space.n
        # cfg.agent.params.action_range = [
        #     float(self.env.action_space.low.min()),
        #     float(self.env.action_space.high.max())
        # ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.obs_shape,self.goal_shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        self.video_recorder = GridVideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs['observation'],obs['desired_goal'], sample=True) # todo: FAlse
                obs, reward, done, _ = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward

            self.video_recorder.save(f'{self.step}.mp4')
            self.logger.log('eval/episode_reward', episode_reward, self.step)
        self.logger.dump(self.step)

    def run_her(self,path_buffer):
        
        #first_obs = path_buffer[0][0]
        #last_obs = path_buffer[-1][0]
        #first_goal = first_obs['achieved_goal']
        #last_goal = last_obs['achieved_goal']
        #goal_changed = np.mean(last_goal - first_goal)**2 > 1e-6

        #if goal_changed:
        for n,ts in enumerate(path_buffer):
            # select goal id
            if self.cfg.her_strat == 'future':
                i = np.random.randint(n,len(path_buffer))
            elif self.cfg.her_strat == 'last':
                i = -1
            new_goal_obs = path_buffer[i][3]
            new_goal = new_goal_obs['achieved_goal']
            # relabel
            obs,action,reward,next_obs,done,done_no_max = ts
            obs['desired_goal'] = new_goal
            next_obs['desired_goal'] = new_goal
            reward = self.env.compute_reward(next_obs['achieved_goal'],new_goal,None)
           # print("REW",reward)
            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                done_no_max)

            

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        path_buffer = []
        last_eval = 0
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step // self.cfg.eval_frequency > last_eval:
                    last_eval += 1
                    self.logger.log('eval/episode', episode, self.step)
                    if self.cfg.save_model:
                        self.agent.save(self.step)
                        self.agent.load(self.step)
                    print("hi evaluating")
                    self.evaluate()
                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

                # her
                if self.cfg.her_iters > 0 and len(path_buffer):
                    for k in range(self.cfg.her_iters):
                        self.run_her(path_buffer)
                path_buffer = []
                        
                            

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs['observation'],obs['desired_goal'], sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)
            path_buffer.append([obs,action,reward,next_obs,done,done_no_max])


            obs = next_obs
            episode_step += 1
            self.step += 1


#workspace = None
@hydra.main(config_path='config/train.yaml', strict=True)
def main(cfg):
   # global workspace
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
