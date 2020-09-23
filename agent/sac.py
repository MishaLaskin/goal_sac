import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

from agent import Agent
import utils

import hydra


class SACAgent(Agent):
    """SAC algorithm."""
    def __init__(self, obs_dim, goal_dim, action_dim, action_range, device,
                 critic_cfg, actor_cfg, discount, init_temperature, alpha_lr,
                 alpha_betas, actor_lr, actor_betas, actor_update_frequency,
                 critic_lr, critic_betas, critic_tau,
                 critic_target_update_frequency, batch_size):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, goal, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        goal = torch.FloatTensor(goal).to(self.device)
        goal = goal.unsqueeze(0)
        dist = self.actor(obs, goal)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
      #  print(action)
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done, achieved_goal, desired_goal, logger, step):
        dist = self.actor(next_obs, desired_goal)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, desired_goal, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, desired_goal, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, achieved_goal, desired_goal, logger, step):
        dist = self.actor(obs, desired_goal)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, desired_goal, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()
        logger.log('train_alpha/loss', alpha_loss, step)
        logger.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done, not_done_no_max, achieved_goal, desired_goal = replay_buffer.sample(
            self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done_no_max, achieved_goal, desired_goal, 
                           logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, achieved_goal, desired_goal, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

    def save(self):
        path = 'saved_models'
        if not os.path.exists(path):
            print('Saving model in:',os.getcwd() + '/' + path)
            os.makedirs(path)
        
        actor_path = path + '/actor.pt'
        critic_path = path + '/critic.pt'
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load(self,path=None):
        if path is None:
            path = 'saved_models'
        actor_path = path + '/actor.pt'
        critic_path = path + '/critic.pt'
        actor_ckpt = torch.load(actor_path)
        critic_ckpt = torch.load(critic_path)
        self.actor.load_state_dict(actor_ckpt)
        self.critic.load_state_dict(critic_ckpt)
        

