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

from .video import VideoRecorder
from .logger import Logger
from .replay_buffer import ReplayBuffer
from . import utils

# import dmc2gym
import hydra


# def make_env(cfg):
#     """Helper function to create dm_control environment"""
#     if cfg.env == 'ball_in_cup_catch':
#         domain_name = 'ball_in_cup'
#         task_name = 'catch'
#     else:
#         domain_name = cfg.env.split('_')[0]
#         task_name = '_'.join(cfg.env.split('_')[1:])
#
#     env = dmc2gym.make(domain_name=domain_name,
#                        task_name=task_name,
#                        seed=cfg.seed,
#                        visualize_reward=True)
#     env.seed(cfg.seed)
#     assert env.action_space.low.min() >= -1
#     assert env.action_space.high.max() <= 1
#
#     return env

class Workspace(object):
    def __init__(self, cfg, env_name, env=None, mujoco=False, goal_idx=0):
        self.work_dir = os.getcwd()
        self.work_dir = os.path.join(self.work_dir, 'data', env_name, f'goal_idx{goal_idx}')
        os.makedirs(self.work_dir, exist_ok=True)
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name,
                             goal_idx=goal_idx)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        if env is None:
            self.env = utils.make_env(cfg)
        else:
            self.env = env

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        print('cfg.agent',cfg.agent)
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0
        self.mujoco = mujoco
        self.goal_idx = goal_idx

    def evaluate(self):
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            trj = []
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                new_obs, reward, done, _ = self.env.step(action)
                trj.append([obs, action, reward, new_obs])
                obs = new_obs
                episode_reward += reward
                self.video_recorder.record(self.env, self.mujoco)
                print(done, action, obs, reward)


            self.video_recorder.save(f'{self.step}.mp4')
            self.logger.log('eval/episode_reward', episode_reward, self.step)
            np.save(os.path.join(self.work_dir, f'trj_eval{episode}_step{self.step}.npy'), np.array(trj))
        self.logger.dump(self.step)

    def evaluate_sample(self, eval_start_num=0):
        for episode in range(self.cfg.num_eval_sample_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            trj = []
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)
                new_obs, reward, done, _ = self.env.step(action)
                trj.append([obs, action, reward, new_obs])
                obs = new_obs
                episode_reward += reward
                self.video_recorder.record(self.env, self.mujoco)
                print(done, action, obs, reward)


            self.video_recorder.save(f'{self.step}.mp4')
            self.logger.log('eval_sample/episode_reward', episode_reward, self.step)
            np.save(os.path.join(self.work_dir, f'trj_evalsample{episode+eval_start_num}_step{self.step}.npy'), np.array(trj))
        self.logger.dump(self.step)

    def run(self):
        #if "cuda" in self.cfg.device:
        #    os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cfg.gpu_id)
        #    print('cuda', os.environ["CUDA_VISIBLE_DEVICES"])
        #import torch
        #import torch.nn as nn
        #import torch.nn.functional as F
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > self.cfg.num_seed_steps and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                    self.evaluate_sample()
                    if self.step > self.cfg.num_seed_steps:
                        self.agent.save_model(output=self.work_dir, step=self.step)
                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

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

            obs = next_obs
            episode_step += 1
            self.step += 1
            print(done, action, obs, reward)

    def run_evaluate(self):
        # Evaluate by loading pre-trained models without training (for generating trajectories)
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:


            # evaluate agent periodically
            if self.step > self.cfg.num_seed_steps and self.step % self.cfg.eval_frequency == 0:
                self.agent.load_model(output=self.work_dir, step=self.step)
                self.evaluate_sample(eval_start_num=self.cfg.eval_start_num)
            self.step += 1



@hydra.main(config_path='config/train.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
