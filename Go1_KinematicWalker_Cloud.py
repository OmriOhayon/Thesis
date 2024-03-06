import time
import matplotlib
import numpy as np
import seaborn as sb
import scipy as scipy
import gymnasium as gym
from gymnasium import Env
from scipy import optimize
from numba import jit, njit
from datetime import datetime
from matplotlib import pyplot as plt
from line_profiler import LineProfiler
from gymnasium.spaces import Box
from memory_profiler import profile as mem_profile
from Go1EnvKinematic import Go1Env, IKine, GenerateTrajectory, MakeVideo, PurePursuit, PIDController, \
                    quat2eul, quat2rot_mj, eul2quat, timer, quat_rot_mj

from mpl_toolkits.mplot3d import Axes3D

### PPO imports
import os
import torch
import random
import argparse
import torch.nn as nn
import torch.optim as optim
from str2bool import str2bool as strtobool
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env


def get_current_reward_func(class_object, name_target):
    for name in dir(class_object):
        if name_target in name:
            return name
    return 'Did not find rewards name'

def render_or_video(env, n_steps, frames_to_skip, block_render, render_from_step, tot_steps, every_x_episode, ep_num):
    if not block_render:
        if ep_num % every_x_episode == 0:
            if n_steps % frames_to_skip == 0:
                if render_from_step <= tot_steps:
                    try:
                        env.render()
                    except:
                        ('Trouble with rendering. Passing and moving on.')
                        pass

def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return

def create_walkers(n, render_first, plot_tracking, xml_file, leg_list, trajectories, leg_id_row, legs_pair, body_length_reducer, args):
    Go1Envs, solvers, pids, planners, controllers, walker_envs = {}, {}, {}, {}, {}, {}

    # PID Gains until 19/2/24:
    # p = np.array([500, 500, 200, 500, 500, 200, 500, 500, 200, 500, 500, 200]) * 1
    # i = np.array([.01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01])
    # d = np.array([.3, .3, 1, .3, .3, 1, .3, .3, 1, .3, .3, 1]) * 15

    p = np.array([500, 500, 200, 500, 500, 200, 500, 500, 200, 500, 500, 200])*1.5*1
    i = np.array([.01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01])
    d = np.array([.7, .8, .6, .7, .8, .6, .7, .8, .6, .7, .8, .6])*12*1.2*1

    for idx in range(1, n + 1):
        env_key = f'env_{idx}'
        Go1Envs[env_key] = Go1Env(xml=xml_file, env_num=idx)

        solver_key = f'my_ikine_{idx}'
        solvers[solver_key] = IKine(env=Go1Envs[env_key])

        pid_key = f'my_pid_{idx}'
        pids[pid_key] = PIDController(P_gain=p, I_gain=i, D_gain=d, env=Go1Envs[env_key])

        planner_key = f'planner_{idx}'
        planners[planner_key] = Planner(env=Go1Envs[env_key], leg_list=leg_list, solver=solvers[solver_key], body_l=0.4*body_length_reducer, body_w=0.1)

        controller_key = f'controller_{idx}'
        controllers[controller_key] = PurePursuit(path=trajectories[0], lookahead_distance=0.3, wheelbase_length=0.3, max_steer_angle=0.10)

        walker_key = f'walker_{idx}'
        if idx == 1:
            if render_first:
                first_env_block_render = False
                first = 1
            else:
                first_env_block_render = True
                first = 1e100
            if plot_tracking:
                plot = True
            else: 
                plot = False
        else:
            first_env_block_render = False
            first = 1e100
            plot = False

        walker_envs[walker_key] = Walker(env=Go1Envs[env_key], planner=planners[planner_key], controller=controllers[controller_key], 
                                         trajectories2d=trajectories, leg_list=leg_list, leg_id_row=leg_id_row, legs_pair=legs_pair, 
                                         block_render=True, render_from_step=first, frames_to_skip_com=5, frames_to_skip_paw=15, # RENDER IS BLOCKED
                                         PID=pids[pid_key], n_traj_for_com=args.num_points_com, plot_tracking=plot, 
                                         n_traj_for_paw=args.num_points_paw)

    return list(walker_envs.values())

class Walker(Env):
    def __init__(self, env, planner, controller, trajectories2d, leg_list, 
                 legs_pair, leg_id_row, render_from_step, 
                 frames_to_skip_com=10, frames_to_skip_paw=10, n_traj_for_com=250, 
                 n_traj_for_paw=250, plot_tracking=False, every_x_ep=1, PID=None, block_render=True):
        super(Walker, self).__init__()

        self.env = env
        self.leg_list = leg_list
        self.legs_pair = legs_pair
        self.leg_id_row = leg_id_row
        self.block_render = block_render
        self.render_from_step = render_from_step
        self.planner = planner
        self.solver = self.planner.solver
        self.controller = controller
        self.trajectories2d = trajectories2d
        self.plot_tracking = plot_tracking
        self.every_x_ep = every_x_ep
        self.pid = PID

        self.n_steps = 0
        self.tot_steps = 0
        self.actions_taken = 0
        self.episode_num = 0
        self.episode_steps = 0
        self.frames_to_skip_com = frames_to_skip_com
        self.frames_to_skip_paw = frames_to_skip_paw
        self.n_traj_for_com = n_traj_for_com
        self.n_traj_for_paw = n_traj_for_paw
        self.controlable_vars = {}
        self.state_memory = []
        self.checkpoint_ep = False
        self.stuck_in_place = False
        self.is_terminal = False

        self.des_angles = np.asarray([0., 0., 0.])
        self.z_com_des = 0.255
        self.paws_dict = {'fr': self.env.fr_calf_num, 'fl': self.env.fl_calf_num, 'rr': self.env.rr_calf_num, 'rl': self.env.rl_calf_num}

        self.gym_env_reqs()
        self.init_variables()
        self.choose_traj()
        self.get_acc()
        self.controller.update_path(new_path=self.trajectory)

        # # ### Reward components up to 8/10/2023:
        # self.target_reward = 10.
        # self.truncated_reward = -10.
        # self.lambda_err = 0.2
        # self.err_norm = 0.
        # self.robot_step_reward = -0.001 ### Minus sign on the prange graph (the later)
        # self.backtrack_reward = -0.01
        # self.ep_return = 0.
        # self.reward_regulator = 1.

        # # ### Reward components 9/10/2023:
        # self.target_reward = 10.
        # self.truncated_reward = -10.
        # self.lambda_err = 0.2
        # self.err_norm = 0.
        # self.robot_step_reward = -0.01 ### Minus sign on the prange graph (the later)
        # self.backtrack_reward = -0.01
        # self.ep_return = 0.
        # self.reward_regulator = 1.

        # ### Reward components up to 13/02/2024:
        self.target_reward = 3.
        self.truncated_reward = -1.
        self.lambda_err = 0.2
        self.err_norm = 0.
        self.robot_step_reward = -0.01 ### Minus sign on the prange graph (the later)
        self.backtrack_reward = -0.01
        self.ep_return = 0.
        self.reward_regulator = 1.

        if self.plot_tracking and self.episode_num % self.every_x_ep == 0:
            plt.ion()
            self.com_pos = plt.figure(figsize=(15, 10))
            self.com_pos_ax = self.com_pos.add_subplot(121, projection='3d')
            self.error_ax = self.com_pos.add_subplot(122)
            plt.show(block=False)

        self.collect_x_loc = [np.asarray([0, 0, 0])]
        self.collect_err = [0]
        self.collect_err_x = [0]
        self.collect_err_y = [0]
        self.collect_closest_distance_err = [0]

        print('Done initializing environment. All 4 paws are touching the surface.')

    def put_legs_on_ground(self):
            des_yaw = 0
            for idx, leg in enumerate(self.leg_list):
                if idx == 1:
                    break
                self.get_up(leg, des_yaw, )
                # self.lift_paw(leg, )
                # self.paw_step(leg, )
            done = 0

    def init_variables(self):
        self.state_cnt = 0
        self.err_norm = 0.
        self.look_ahead = self.controller.lookahead_distance
        self.dx_action = 0.03
        self.dy_action = 0.16029
        self.dz_action = 0.245
        self.des_yaw = 0.
        self.ep_return = 0.
        self.episode_steps = 0
        self.terminated = False
        self.is_terminal = False
        self.stuck_in_place = False
        self.sim_diverged = False
        self.got_to_goal = 0
        self.try_cnt = 0
        self.collect_x_loc = [np.asarray([0, 0, 0])]
        self.collect_err = [0]
        self.collect_err_x = [0]
        self.collect_err_y = [0]
        self.collect_closest_distance_err = [0]

    def gym_env_reqs(self):
        self.observation_space = Box(low=-np.Inf, high=np.Inf, shape=(32,), dtype=np.float32)
        
        # low_array = np.asarray([0.04, 0.1729, 0.25, 0.1], dtype=np.float32)
        # high_array = np.asarray([0.05, 0.1829, 0.3, 0.8], dtype=np.float32)

        # ### Up to 29/8/2023, 17:30
        # low_array = np.asarray([0.025, 0.1129, 0.24, 0.5], dtype=np.float32)
        # high_array = np.asarray([0.1, 0.209, 0.32, 3], dtype=np.float32)
        # ###
        # ### Up to 04/9/2023, 22:30
        # low_array = np.asarray([0.03, 0.1129, 0.24, 0.5], dtype=np.float32)
        # high_array = np.asarray([0.12, 0.209, 0.32, 3], dtype=np.float32)
        # ###
        ### 
        # ### Up to 18/9/2023, 20:08
        # low_array = np.asarray([0.025, 0.1229, 0.28, 0.5], dtype=np.float32)
        # high_array = np.asarray([0.10, 0.1909, 0.32, 3], dtype=np.float32)
        # ###
        ### Up to 18/9/2023, 20:08
        low_array = np.asarray([0.005, 0.1229, 0.26, 0.5], dtype=np.float32)
        high_array = np.asarray([0.04, 0.1909, 0.32, 3], dtype=np.float32)
        ###

        # low_array = np.asarray([0.02, 0.1029, 0.22, 0.5], dtype=np.float32)
        # high_array = np.asarray([0.15, 0.2209, 0.34, 3], dtype=np.float32)
        self.action_space = Box(low=low_array, high=high_array, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.actions_taken = 0

        if self.state_cnt >= 70 and self.is_terminal == False:
            self.checkpoint_ep = True
            self.info['final_info']['Checkpoint'] = self.checkpoint_ep
            print(f'Bootstarping env num {self.env.env_num} from position: {self.env.robot_data.qpos[0:3]}')
            state_to_start_from = self.state_memory[self.state_cnt-20]
            state_to_start_from[2] += 0.05
            self.env.reset_sim(init_pos=state_to_start_from, booted=True)
            self.choose_traj(self.trajectory_index)
        else:
            self.checkpoint_ep = False
            if self.n_steps >= 2:
                self.info['final_info']['Checkpoint'] = self.checkpoint_ep
            self.env.init_pos[0:2] += np.random.uniform(-0.05, 0.05, 2)
            self.env.reset_sim(init_pos=self.env.init_pos)
            self.choose_traj()
            self.put_legs_on_ground()
        
        self.init_variables()
        self.state_memory = []
        self.controller.update_path(new_path=self.trajectory)
        init_q_pos = self.env.get_qpos_without_quat() 
        init_q_pos[3:6] *= (10/(np.pi*2)) # The body angles are prescaled differently
        
        # self.init_obs = np.asarray([*init_q_pos, 
        #                        *self.env.get_4_paws(), 
        #                        self.err_norm, 
        #                        self.look_ahead]).reshape(-1,).astype(np.float32) / 10 # Pre-scale the observation
        
        self.init_obs = np.asarray([*init_q_pos, 
                               *self.env.get_4_paws(), 
                               0., 
                               self.look_ahead]).reshape(-1,).astype(np.float32) / 10 # Pre-scale the observation

        self.init_obs = np.around(self.init_obs, 4)

        self.init_info = {'obs': self.init_obs, 'env_num': self.env.env_num}

        return self.init_obs, self.init_info

    def choose_traj(self, traj_idx=None):
        if traj_idx == None:
            self.trajectory_index = np.random.randint(0, len(self.trajectories2d))
        else: self.trajectory_index = traj_idx

        self.trajectory = self.trajectories2d[self.trajectory_index]
        
        # ### Get the trajectory's length:
        self.total_traj_len = self.trajectory_length(traj=self.trajectory)

    def trajectory_length(self, traj):
        traj_diff = np.diff(traj, axis=0)
        distances = np.linalg.norm(traj_diff, axis=1)
        return np.sum(distances)

    def effective_percentage(self, current_point=None):
        traj = self.trajectory
        closest_point_idx = self.controller.last_closest_idx
        closest_point = traj[closest_point_idx]
        sub_traj = traj[:closest_point_idx+2]

        length_to_closest = self.trajectory_length(sub_traj)
        distance_to_current = np.linalg.norm(current_point - closest_point)
        # distance_to_current = self.closest_on_path[:]

        effective_distance = length_to_closest - distance_to_current
        effective_percentage = (effective_distance / self.total_traj_len)
        clamped_eff_per = min(max(effective_percentage, -1.), 1.)
        
        return clamped_eff_per * 0.5

    # def append_state_deprecated(self):
    #     if len(self.state_memory) >= 30:
    #         self.state_memory.pop(0)
    #         self.state_memory.append([*self.env.robot_data.qpos[:]])
    #     else:
    #         self.state_memory.append([*self.env.robot_data.qpos[:]])
    #     self.state_cnt += 1

    def append_state(self):
        self.state_memory.append([*self.env.robot_data.qpos[:]])
        self.state_cnt += 1

    def step(self, action):
        # action = action.flatten()
        leg = self.leg_list[self.actions_taken % 4]
        self.actions_taken += 1
        self.episode_steps += 1
        
        self.dx_action = action[0]
        self.dy_action = action[1]
        self.dz_action = action[2]
        self.controller.lookahead_distance = action[3]

        # print(self.dx_action,
        # self.dy_action,
        # self.dz_action,
        # self.controller.lookahead_distance)

        delta = self.controller.compute_steering_angle(self.env.robot_data.qpos[:2], self.des_angles[-1])
        self.des_yaw += delta

        # ### Use the action:
        self.move_com(leg, self.des_yaw, )
        self.lift_paw(leg, )
        self.paw_step(leg, )

        self.append_state()
        if self.state_cnt >= 16:
            self.check_advancement()

        self.check_divergences()

        now_xy_pos = self.env.robot_data.qpos[:2]
        self.err_norm = np.linalg.norm(now_xy_pos - self.controller.path[self.controller.last_closest_idx])
        self.closest_on_path, self.closest_distance_err = self.controller.find_closest_on_path(now_xy_pos)
        self.effective_percentage_on_traj = self.effective_percentage(current_point=now_xy_pos)

        # ### Collect data:
        self.step_reward = self.get_reward_13_02_1st()
        self.ep_return += self.step_reward
        # print('Step reward:', self.step_reward)
        obs = self.get_obs()
        
        if self.plot_tracking and self.episode_num % self.every_x_ep == 0:
            self.plot_com_pos()
            self.plot_pos_error()

        if self.is_terminal or self.env.done:
            self.terminated = True

        self.info = {'obs': obs, 'reward': np.asarray((self.step_reward)), 
                     'truncated': np.asarray((self.terminated)), 'terminated': np.asarray((self.terminated))}
        
        episodic_return = self.ep_return*1
        if self.terminated:
            self.episode_num += 1
            self.info = {'obs': obs, 'reward': np.asarray((self.step_reward)), 
                     'truncated': np.asarray((self.terminated)), 'terminated': np.asarray((self.terminated)), 
                     'final_info': {'episode': {'ep_num': self.episode_num, 
                                                'r': episodic_return, 
                                                'l': self.episode_steps, 
                                                'env_num': self.env.env_num, 
                                                'Checkpoint': self.checkpoint_ep,
                                                'got_to_goal': self.got_to_goal}}}
            self.reset()

        obs = np.around(obs, 4)
        return (obs, np.asarray((self.step_reward)), 
                np.asarray((self.terminated)), np.asarray((self.terminated)), 
                self.info)

    def get_reward_21_08_1st(self, ):
        if self.env.robot_data.ncon >= 12 or self.stuck_in_place or self.sim_diverged:
            self.env.done = True
        # forces = self.env.get_contact_forces()
        # iani_forces = self.env.get_force_sensors()

        step_reward = 0.
        self.is_terminal = self.check_if_terminal()

        if self.env.done:
            step_reward += self.truncated_reward
        
        if self.is_terminal:
            step_reward += self.target_reward

        if self.backtrack:
            step_reward += self.backtrack_reward

        # err_reward = -np.tanh(self.lambda_err*self.err_norm)*2 + 1
        err_reward = np.exp(-self.err_norm) * 0.5

        step_reward += self.robot_step_reward
        # step_reward += err_reward
        step_reward += self.effective_percentage_on_traj * 0.1

        return step_reward / self.reward_regulator

    def get_reward_09_10_1st(self, ):
        if self.env.robot_data.ncon >= 12 or self.stuck_in_place or self.sim_diverged:
            self.env.done = True
        # forces = self.env.get_contact_forces()
        # iani_forces = self.env.get_force_sensors()

        step_reward = 0.
        self.is_terminal = self.check_if_terminal()

        if self.env.done:
            step_reward += self.truncated_reward
        
        if self.is_terminal:
            step_reward += self.target_reward

        if self.backtrack:
            step_reward += self.backtrack_reward

        # err_reward = -np.tanh(self.lambda_err*self.err_norm)*2 + 1
        # err_reward = np.exp(-self.err_norm) * 0.5

        step_reward += self.robot_step_reward
        # step_reward += err_reward
        step_reward += self.effective_percentage_on_traj * 0.1

        return step_reward / self.reward_regulator

    def get_reward_13_02_1st(self, ):
        if self.stuck_in_place or self.sim_diverged:
            self.env.done = True
        # forces = self.env.get_contact_forces()
        # iani_forces = self.env.get_force_sensors()

        step_reward = 0.
        self.is_terminal = self.check_if_terminal()

        if self.env.done:
            step_reward += self.truncated_reward
        
        if self.is_terminal:
            step_reward += self.target_reward

        if self.backtrack:
            step_reward += self.backtrack_reward

        # err_reward = -np.tanh(self.lambda_err*self.err_norm)*2 + 1
        err_reward = np.exp(-self.err_norm) * 0.5

        step_reward += self.robot_step_reward
        # step_reward += err_reward
        step_reward += self.effective_percentage_on_traj * 1.

        return step_reward / self.reward_regulator

    def check_if_terminal(self, ):
        if self.controller.last_closest_idx >= self.controller.path_len - 3 and not self.sim_diverged:
            print(f'\n\nFinished trajectory on env {self.env.env_num} and traj idx {self.trajectory_index}. \nFinal state: {self.env.robot_data.qpos[:3]}\n\n')
            self.got_to_goal = 1
            return True
        else: 
            return False

    def check_advancement(self, ):
        last_x_states = np.asarray(self.state_memory[-15:])[:, :2]
        diff_last_states = np.sum(np.linalg.norm(np.diff(last_x_states, axis=0), axis=1))
        if diff_last_states <= 0.2:
            self.stuck_in_place = True

    def check_divergences(self, ):
        z_pos = self.env.robot_data.qpos[2]
        if z_pos >= 4. or z_pos <= -0.01:
            self.sim_diverged = True

    def get_obs(self, ):
        q_pos = self.env.get_qpos_without_quat()
        q_pos[3:6] *= (10/(np.pi*2))
        # obs = np.asarray([*q_pos, 
        #                 #   *self.env.robot_data.qpos, 
        #                        *self.env.get_4_paws(), 
        #                        self.err_norm, 
        #                        self.controller.lookahead_distance]).reshape(-1,).astype(np.float32) / 10
        
        obs = np.asarray([*q_pos, 
                        #   *self.env.robot_data.qpos, 
                               *self.env.get_4_paws(), 
                               self.effective_percentage_on_traj*10, 
                               self.controller.lookahead_distance]).reshape(-1,).astype(np.float32) / 10
        
        return obs

    def move_com(self, leg_id: str, des_yaw: float, z_com_des=0.225):
        self.des_yaw = des_yaw
        des_angles = self.planner.get_des_angles(self.des_yaw)
        z_com_des = self.dz_action
        # z_com_des = self.planner.get_final_z_pos(z_com_des)
        
        self.des_angles = np.asarray(des_angles)
        line_com = self.env.get_polygon_line(stepping_leg=leg_id, n=self.n_traj_for_com, z_des=z_com_des).transpose()
        self.n_steps = 0
        self.backtrack = 0
        self.reverse_step = 0
        good_idx = self.n_steps
        custom_qpos = self.planner.generate_body_traj(line_com=line_com, des_angles=self.des_angles)
        append_qpos = []
        custom_qpos = custom_qpos[10:]
        # custom_qpos = np.concatenate((np.tile(custom_qpos[0], (50, 1)), custom_qpos), axis=0)
        custom_qpos = np.concatenate((custom_qpos, np.tile(custom_qpos[-1:], (20, 1))), axis=0)
        for nn, pos in enumerate(custom_qpos):
            des_vel = self.vel_com[nn].copy()
            if nn/len(custom_qpos) >= 0.7:
                des_vel += -0.1

            render_or_video_params = [self.env, self.n_steps, self.frames_to_skip_com, 
                                    self.block_render, self.render_from_step, self.tot_steps,
                                    self.every_x_ep, self.episode_num]
            self.tot_steps += 1
            
            action = self.pid.calc_signal(pos[7:], des_vel)
            self.env.step_sim(action)
            
            append_qpos.append([*self.env.robot_data.qpos[:]])
            render_or_video(*render_or_video_params)
            
            good_idx = self.n_steps
            self.n_steps += 1

            if self.backtrack:
                self.planner.backtrack_steps(append_qpos, render_or_video_params, action, self.n_steps, self.reverse_step, pid=self.pid)
                break

        if not self.reverse_step:
            action = self.pid.calc_signal(append_qpos[good_idx][7:])
            self.env.step_sim(action)
            
        # action = np.zeros((12, ))
        # for i in range(5):
        #     # action = self.pid.calc_signal(append_qpos[nn-1][7:])
        #     self.env.step_sim(action)

        if not self.block_render:
            if self.render_from_step <= self.tot_steps:
                try:
                    self.env.render()
                except:
                    pass

    def get_up(self, leg_id: str, des_yaw: float, z_com_des=0.235):
        self.des_yaw = des_yaw
        des_angles = self.planner.get_des_angles(self.des_yaw)
        z_com_des = 0.245
        # z_com_des = self.planner.get_final_z_pos(z_com_des)
        
        self.des_angles = np.asarray(des_angles)
        line_com = self.env.get_polygon_line(stepping_leg=leg_id, n=self.n_traj_for_com, z_des=z_com_des).transpose()
        self.n_steps = 0
        self.backtrack = 0
        self.reverse_step = 0
        good_idx = self.n_steps
        custom_qpos = self.planner.generate_body_traj(line_com=line_com, des_angles=self.des_angles)
        append_qpos = []
        custom_qpos = np.concatenate((np.tile(custom_qpos[0], (50, 1)), custom_qpos), axis=0)
        custom_qpos = np.concatenate((custom_qpos, np.tile(custom_qpos[-1:], (20, 1))), axis=0)
        des_vel = np.zeros((12, ))
        for nn, pos in enumerate(custom_qpos):
            # des_vel = self.vel_com[nn].copy()

            action = self.pid.calc_signal(pos[7:], des_vel)
            self.env.step_sim(action/6)
            
        action = np.zeros((12, ))
        for i in range(5):
            # action = self.pid.calc_signal(append_qpos[nn-1][7:])
            self.env.step_sim(action)

    def paw_step(self, leg_id, circle_move=False, trapezoid=False, climb=False):
        n = self.n_traj_for_paw
        self.n_steps = 0
        traj_idx = 0
        done_step = 0
        leg_idx = self.leg_id_row[leg_id]
        des_pos = self.env.robot_data.qpos.copy()
        if not circle_move:
            _, _, dz_max, _ = self.planner.get_deltas(leg_id=leg_id, des_angles=self.des_angles)
            dx_size, dy_size, self.z_com_des = self.dx_action, self.dy_action, self.dz_action
            placement = self.planner.get_paw_placement(leg_id=leg_id, delta_x_size=dx_size, delta_y_size=dy_size)
            placement[-1] = dz_max
            x_dist, y_dist, z_height = placement

            if trapezoid == False:
                des_points = self.planner.generate_step_traj(leg=leg_id, traj_type='parabola', delta_y=y_dist, delta_x=x_dist, delta_z=z_height, n=n)
            else:
                des_points = self.planner.generate_step_traj(leg=leg_id, traj_type='trapezoid', delta_y=y_dist, delta_x=x_dist, delta_z=z_height, n=n)
            
            if self.planner.no_traj == True:
                self.planner.no_traj = False
                return -1
        else:
            leg_pos = self.env.get_site_pos(leg_id.upper()+"_touch_sensor")
            des_points = self.planner.circle_step(leg_pos_0=leg_pos, radius=0.02)

        append_qpos = []
        last_coords = np.zeros((3, ))
        self.reverse_step = 0
        self.finished_step = 0
        self.backtrack = 0
        now_vel = self.vel_paw.copy()
        for nn, point in enumerate(des_points):
            des_vel = now_vel[nn]

            render_or_video_params = [self.env, self.n_steps, self.frames_to_skip_paw, 
                                    self.block_render, self.render_from_step, self.tot_steps,
                                    self.every_x_ep, self.episode_num]
            self.tot_steps += 1
            ### Solve i Kine:
            last_coords[:] = des_pos[leg_idx:leg_idx+3]
            try:
                my_coords = self.solver.solve(leg=leg_id.lower(), leg_idx=leg_idx, theta=last_coords, x_des=point)
                # my_coords = solve(env=self.env, leg=leg_id.lower(), leg_idx=leg_idx, theta=last_coords, x_des=point)
            except:
                self.env.done = True
                break
            
            if self.planner.check_singularity(leg_id_list=[leg_id]):
                my_coords[:] = last_coords[:]
                self.backtrack = 1
            
            des_pos[leg_idx:leg_idx+3] = my_coords[0:3]
            action = self.pid.calc_signal(des_pos[7:], des_vel)
            action[-1] *= 2
            action[-4] *= 2
            self.env.step_sim(action)
            append_qpos.append([*self.env.robot_data.qpos[:]])

            self.n_steps += 1
            traj_idx += 1
            
            render_or_video(*render_or_video_params)

            if self.planner.check_paw_touch(leg_id, traj_idx, self.n_steps, self.finished_step):
                # des_vel[:vel_leg_idx] *= 0
                paw_force, normal = self.env.get_paw_contact_force(leg_id=leg_id)
                if self.env.check_force_direction(force=normal) and not circle_move:
                    self.try_cnt += 1
                    self.planner.increase_dx += 0.02
                    if self.try_cnt >= 3:
                        # self.lift_paw(leg_id=leg_id, put_leg_down=True)
                        break
                    self.paw_step(leg_id=leg_id, circle_move=True)
                    self.move_com(leg_id=leg_id, des_yaw=self.des_yaw)
                    self.lift_paw(leg_id=leg_id)
                    self.paw_step(leg_id=leg_id, trapezoid=False)
                else:
                    del des_pos
                    self.try_cnt = 0
                    self.planner.increase_dx = 0 
                    done_step = 1
                    # break

            if done_step:
                for i in range(5):
                    action = self.pid.calc_signal(append_qpos[nn-1][7:])
                    self.env.step_sim(action*0)
                break

            if self.backtrack:
                self.planner.backtrack_steps(append_qpos, render_or_video_params, 
                                                action, self.n_steps, self.reverse_step, pid=self.pid)
                break
            
        # if (not self.reverse_step) and self.finished_step:
        #     good_idx = traj_idx - 2
        # #     good_idx = traj_idx - 1
        # #     # self.env.robot_data.qpos[7:] = append_qpos[good_idx][7:]
            
        # #     # self.env.robot_data.qpos[:] = append_qpos[good_idx][:]
        #     action = self.pid.calc_signal(append_qpos[good_idx][7:])
        #     self.env.step_sim(action)

        # self.env.step_sim(self.env.get_torque(self.env.robot_data.qpos, self.env.robot_data))
        # self.env.step_sim(action)

        if not self.block_render:
            if self.render_from_step <= self.tot_steps:
                try:
                    self.env.render()
                except:
                    pass
        done = 0
        # print("Done Step With", leg_id, "Leg.")

    def lift_paw(self, leg_id, put_leg_down=False):
        n = 25
        self.n_steps = 0
        traj_idx = 0
        leg_idx = self.leg_id_row[leg_id]
        vel_leg_idx = leg_idx - 7
        des_pos = self.env.robot_data.qpos.copy()
        self.now_z_l = self.env.get_site_pos(leg_id+"_touch_sensor")[2]
        self.now_z_h = self.env.get_joint_anchor(leg_id+"_hip_joint")[2]
        z_height_divider = 7
        z = (self.now_z_h - self.now_z_l)/z_height_divider
        # des_points = self.planner.generate_step_traj(leg=leg_id, traj_type='line', delta_y=0, delta_x=0, delta_z=0.025, n=n)
        if not put_leg_down:
            self.lift_des_points = self.planner.generate_step_traj(leg=leg_id, traj_type='line', delta_y=0, delta_x=0, delta_z=z, n=n)
        else:
            self.lift_des_points = self.lift_des_points[::-1, ...]

        append_qpos = []
        last_coords = np.zeros((3, ))
        self.reverse_step = 0
        self.finished_step = 0
        self.backtrack = 0

        now_vel = self.vel_paw.copy()
        now_vel[:vel_leg_idx] *= 0
        now_vel[vel_leg_idx+3:] *= 0
        for nn, point in enumerate(self.lift_des_points):
            render_or_video_params = [self.env, self.n_steps, self.frames_to_skip_paw, 
                                    self.block_render, self.render_from_step, self.tot_steps,
                                    self.every_x_ep, self.episode_num]
            self.tot_steps += 1
            ### Solve i Kine:
            last_coords[:] = des_pos[leg_idx:leg_idx+3]
            try:
                my_coords = self.solver.solve(leg=leg_id.lower(), leg_idx=leg_idx, theta=last_coords, x_des=point)
                # my_coords = solve(env=self.env, leg=leg_id.lower(), leg_idx=leg_idx, theta=last_coords, x_des=point)
            except:
                self.env.done = True
                break

            des_pos[leg_idx:leg_idx+3] = my_coords[0:3]

            action = self.pid.calc_signal(des_pos[7:], self.vel_paw[nn])
            self.env.step_sim(action)
            append_qpos.append([*self.env.robot_data.qpos[:]])

            self.n_steps += 1
            traj_idx += 1
            
            render_or_video(*render_or_video_params)

            if self.planner.check_paw_touch(leg_id, traj_idx, self.n_steps, self.finished_step):
                action = np.zeros((12, ))
                for i in range(5):
                    self.env.step_sim(action)
                break

            if self.backtrack:
                # self.planner.backtrack_steps(append_qpos, render_or_video_params, 
                #                                 self.action, self.n_steps, self.reverse_step, pid=self.pid)
                break

        # action = np.zeros((12, ))
        # for i in range(5):
        #     self.env.step_sim(action)

        if not self.block_render:
            if self.render_from_step <= self.tot_steps:
                try:
                    self.env.render()
                except:
                    pass

        # print("Done Paw Lift With", leg_id, "Leg.")

    def get_acc(self, des_vel_com=None, des_vel_paw=None):
        des_vel_com = np.asarray([0.033, 0.053, 0.083, 0.033, 0.053, 0.083, 0.033, 0.053, 0.083, 0.033, 0.053, 0.083])/10
        # des_vel_com = np.asarray([0.033, 0.053, 0.083, 0.033, 0.053, 0.083, 0.033, 0.053, 0.083, 0.033, 0.053, 0.083])/100*0
        des_vel_paw = np.asarray([0.033, 0.053, 0.013, 0.033, 0.053, 0.013, 0.033, 0.053, 0.013, 0.033, 0.053, 0.013])/36*0

        n_com = self.n_traj_for_com + 10
        acc_phase_length = n_com // 2  # 25% of the trajectory for acceleration
        dec_phase_length = acc_phase_length  # Same for deceleration
        const_phase_length = n_com - acc_phase_length - dec_phase_length  # Remaining for constant velocity
        velocity_profiles_com = np.zeros((12, n_com))
        for i in range(12):
            acc_phase = np.linspace(0, des_vel_com[i], acc_phase_length)
            const_phase = np.full(const_phase_length, des_vel_com[i])
            dec_phase = np.linspace(des_vel_com[i], 0, dec_phase_length)
            velocity_profiles_com[i] = np.concatenate((acc_phase, const_phase, dec_phase))

        velocity_profiles_com = velocity_profiles_com.T

        n_paw = self.n_traj_for_paw*2
        acc_phase_length = n_paw // 2  # 25% of the trajectory for acceleration
        dec_phase_length = acc_phase_length  # Same for deceleration
        const_phase_length = n_paw - acc_phase_length - dec_phase_length  # Remaining for constant velocity
        velocity_profiles_paw = np.zeros((12, n_paw))
        for i in range(12):
            acc_phase = np.linspace(0, des_vel_paw[i], acc_phase_length)
            const_phase = np.full(const_phase_length, des_vel_paw[i])
            dec_phase = np.linspace(des_vel_paw[i], 0, dec_phase_length)
            velocity_profiles_paw[i] = np.concatenate((acc_phase, const_phase, dec_phase))

        velocity_profiles_paw = velocity_profiles_paw.T

        self.vel_com = velocity_profiles_com
        self.vel_paw = velocity_profiles_paw
        pass

    def plot_com_pos(self, ):
        self.com_pos_ax.clear()
        self.com_pos_ax.set_title('Center of Mass 3D Position: \n Lookahead value:{:.3f}'.format(self.controller.lookahead_distance))
        self.com_pos_ax.set_xlabel('X')
        self.com_pos_ax.set_ylabel('Y')
        self.com_pos_ax.set_zlabel('Z')

        self.collect_x_loc.append(self.env.robot_data.qpos[:3].copy())
        x = np.asarray(self.collect_x_loc)[:, 0]
        y = np.asarray(self.collect_x_loc)[:, 1]
        z = np.asarray(self.collect_x_loc)[:, 2]
        if self.trajectory is not None:
            self.com_pos_ax.scatter(self.trajectory[:, 0], self.trajectory[:, 1], label='Trajectory to follow')
            self.com_pos_ax.scatter(*self.controller.lookahead_point, facecolors='red', s=72, label='Current lookahead point')
            self.com_pos_ax.scatter(*self.closest_on_path, facecolors='yellow', s=72, label='Min. distance of CoM from path')
            
        self.com_pos_ax.scatter(x, y, z)
        self.com_pos_ax.legend()
        self.com_pos.canvas.draw()
        mypause(0.005)
        # plt.pause(0.005)

    def plot_pos_error(self, ):
        self.error_ax.clear()
        self.error_ax.set_title('Center of Mass L2 Distance From Current Lookahead Point: ')
        self.error_ax.set_xlabel('Center of Mass Step')
        self.error_ax.set_ylabel('Position Error [m]')

        self.collect_err.append(self.err_norm)
        self.collect_err_x.append(self.controller.lookahead_point[0] - self.env.robot_data.qpos[0])
        self.collect_err_y.append(self.controller.lookahead_point[1] - self.env.robot_data.qpos[1])
        self.collect_closest_distance_err.append(self.closest_distance_err)
        self.error_ax.plot(list(range(len(self.collect_err))), self.collect_err, label='L2 distance')
        self.error_ax.plot(list(range(len(self.collect_err_x))), self.collect_err_x, label='X axis error')
        self.error_ax.plot(list(range(len(self.collect_err_y))), self.collect_err_y, label='Y axis error')
        self.error_ax.plot(list(range(len(self.collect_closest_distance_err))), self.collect_closest_distance_err, label='Min. distance from path')
        self.error_ax.legend()
        self.com_pos.canvas.draw()
        mypause(0.005)
        # plt.pause(0.005)
        
class Planner():
    def __init__(self, env, leg_list, solver: IKine, body_l=0.4, body_w=0.1):
        self.env = env
        self.leg_list = leg_list
        self.body_l = body_l
        self.body_w = body_w
        self.solver = solver
        self.n_circle_points = 100
        self.points = np.linspace(0, 1, self.n_circle_points)
        self.two_pi = np.linspace(np.pi/2, -5/2*np.pi, 50)
        self.no_traj = False # Changes to True if no parabolic trajectory could be generated
        self.after_circle = False # Changes to True after 
        self.increase_dx = 0
        self.pitch_mul = 1

    def get_dofs_idx(self, leg_id: str=None):
        """
            Return indices of the actoators's dofs.
                nv: Number of degrees of freedom.
                leg_id: ID of paw to move in case of step. None if movement is for CoM.

        """
        nv = self.env.robot.nv
        trunk_dofs = range(6)
        body_dofs = range(6, nv)

        if leg_id == None:
            return trunk_dofs, body_dofs, None
        else: 
            # Get all joint names.
            joint_names = [self.env.robot.joint(i).name for i in range(self.env.robot.njnt)]
            # Get indices into relevant sets of joints.
            moving_leg_dofs = np.array([
                self.env.robot.joint(name).dofadr[0]
                for name in joint_names
                if leg_id in name])
            return trunk_dofs, body_dofs, moving_leg_dofs

    def check_singularity(self, leg_id_list: list,):
        for leg in leg_id_list:
            paw_pos = self.env.get_site_pos(leg+"_touch_sensor")
            hip_pos = self.env.get_joint_anchor(leg+"_hip_joint")
            diff = paw_pos - hip_pos
            norm = np.sqrt(np.dot(diff, diff))

            if norm >= 0.40500:
                return True

    def check_touch_or_singularity(self, n_steps):
            touching_, contacts_ = self.env.get_touching_parts()
            singular_ = self.check_singularity(leg_id_list=self.leg_list)
            # print('\n*******', touching_)
            if any('thigh' in part or 'trunk' in part or 'hip' in part for part in touching_) or singular_:
                good_idx = n_steps-1
                backtrack = 1
            else: 
                good_idx = n_steps
                backtrack = 0

            return good_idx, backtrack
    
    def backtrack_steps(self, pos_list, render_or_video_params, action, n_steps, reverse_step, pid):
                    back_pos = pos_list[:-2].copy()
                    back_pos.reverse()
                    for qpos in back_pos[:30]:
                        err = qpos[7:] - self.env.robot_data.qpos[7:]
                        action = pid.calc_signal(err)
                        self.env.step_sim(action)
                        render_or_video(*render_or_video_params)
                        n_steps += 1
                    reverse_step = True
    
    def check_paw_touch(self, leg_id, traj_idx, n_steps, finished_step):
                tounching_paws = self.env.count_touching_paws()[2]
                if leg_id in tounching_paws and n_steps > 20:
                    good_idx = traj_idx-1
                    finished_step = 1
                    return True
    
    def get_des_angles(self, yaw: float=0.0):
        fr = self.env.get_site_pos("FR_touch_sensor")
        fl = self.env.get_site_pos("FL_touch_sensor")
        rr = self.env.get_site_pos("RR_touch_sensor")
        rl = self.env.get_site_pos("RL_touch_sensor")
        
        quat = self.env.robot_data.qpos[3:7]
        
        fr_body = quat_rot_mj(vec=fr, q=quat, center_of_rotation=self.env.robot_data.qpos[:3])
        fl_body = quat_rot_mj(vec=fl, q=quat, center_of_rotation=self.env.robot_data.qpos[:3])
        rr_body = quat_rot_mj(vec=rr, q=quat, center_of_rotation=self.env.robot_data.qpos[:3])
        rl_body = quat_rot_mj(vec=rl, q=quat, center_of_rotation=self.env.robot_data.qpos[:3])

        z_f = (fl[2]+fr[2])/2
        z_r = (rl[2]+rr[2])/2
        x_f = (fl[0]+fr[0])/2
        x_r = (rl[0]+rr[0])/2

        z_f_body = (fl_body[2]+fr_body[2])/2
        z_r_body = (rl_body[2]+rr_body[2])/2
        x_f_body = (fl_body[0]+fr_body[0])/2
        x_r_body = (rl_body[0]+rr_body[0])/2

        z_left = (fl_body[2]+rl_body[2])/2
        z_right = (fr_body[2]+rr_body[2])/2
        y_left = (fl_body[1]+rl_body[1])/2
        y_right = (fr_body[1]+rr_body[1])/2
        
        des_pitch = np.arctan((z_f - z_r)/(self.body_l*1 + x_f_body - x_r_body))
        des_roll = np.arctan((z_left - z_right)/(self.body_w*1 + y_left - y_right)) * 1

        # print('Roll: {}, Pitch: {}, Yaw: {}'.format(des_roll, -des_pitch, yaw))
        return [des_roll/1, -des_pitch*self.pitch_mul, yaw]

    # def generate_body_traj(self, line_com: np.ndarray, des_angles: list=[0, 0, 0]) -> np.array:
    #     """
    #     Docs
    #     """
    #     fr_des = self.env.get_site_pos("FR_touch_sensor")
    #     fl_des = self.env.get_site_pos("FL_touch_sensor")
    #     rr_des = self.env.get_site_pos("RR_touch_sensor")
    #     rl_des = self.env.get_site_pos("RL_touch_sensor")

    #     des_qpos = np.zeros((19, len(line_com)))
    #     des_array = [fr_des, fl_des, rr_des, rl_des]
    #     angles, trunk_angles = self.solver.get_angles(com_des_arr=line_com, des_array=des_array, des_angles=des_angles)

    #     des_qpos[:3, :], des_qpos[7:, :], des_qpos[3:7, :] = line_com.T, np.asarray(angles).T, trunk_angles.T
    #     des_qpos = des_qpos.T

    #     return des_qpos
    
    def generate_body_traj(self, line_com: np.ndarray, des_angles: list=[0, 0, 0]) -> np.array:
        """
        Docs
        """
        fr_des = self.env.get_site_pos("FR_touch_sensor")
        fl_des = self.env.get_site_pos("FL_touch_sensor")
        rr_des = self.env.get_site_pos("RR_touch_sensor")
        rl_des = self.env.get_site_pos("RL_touch_sensor")

        des_qpos = np.zeros((19, len(line_com)))
        des_array = [fr_des, fl_des, rr_des, rl_des]
        angles, trunk_angles = self.solver.get_angles(com_des_arr=line_com, des_array=des_array, des_angles=des_angles)

        des_qpos[:3, :], des_qpos[7:, :], des_qpos[3:7, :] = line_com.T, np.asarray(angles).T, trunk_angles.T
        des_qpos = des_qpos.T

        return des_qpos
    
    def get_paw_placement(self, leg_id: str, delta_x_size: float, delta_y_size: float):
        
        paw_pos = self.env.get_site_pos(leg_id+"_touch_sensor")
        center_of_rot = self.env.robot_data.qpos[0:3]
        quat = self.env.robot_data.qpos[3:7]

        # new_rot = quat2rot_mj(quat)
        # new_rot_x = new_rot[:2, 0]
        # new_rot_y = new_rot[:2, 1]
        # normed_x, normed_y = new_rot_x/np.linalg.norm(new_rot_x), new_rot_y/np.linalg.norm(new_rot_y) 
        # final_rot = np.column_stack((normed_x, normed_y, [[0], [0]]))
        # final_rot = np.row_stack((final_rot, [0, 0, 1]))
        # # print(normed_x, normed_y)

        paw_pos_body = quat_rot_mj(paw_pos, quat, center_of_rotation=center_of_rot)
        # paw_pos_body = final_rot @ (paw_pos - center_of_rot) + center_of_rot

        hip_pos = self.env.get_joint_anchor(leg_id+"_hip_joint")
        hip_pos_body = quat_rot_mj(hip_pos, quat, center_of_rotation=center_of_rot)
        # hip_pos_body = final_rot @ (hip_pos - center_of_rot) + center_of_rot


        diff_hip_body = paw_pos_body - hip_pos_body
        
        if leg_id == "FR" or leg_id == "RR":
            dy = -1
        else: dy = 1
        
        # delta_x_size = 0
        if self.after_circle == True:
            self.after_circle = False
            delta_x = delta_x_size - diff_hip_body[0] + 0.08 + self.increase_dx
        else:
            delta_x = delta_x_size - diff_hip_body[0] + 0.01
        delta_y = delta_y_size*dy - diff_hip_body[1]

        new_paw_pos = quat_rot_mj(paw_pos_body + [delta_x, delta_y, 0], quat, center_of_rotation=center_of_rot, inv=True)
        # new_paw_pos = final_rot.T @ (paw_pos_body + [delta_x, delta_y, 0] - center_of_rot) + center_of_rot

        delta_x, delta_y, _ = new_paw_pos - paw_pos
        # if delta_x < 0.009:
        #     delta_x = 0.01
        delta_placement = [delta_x, delta_y, 0]
        
        return delta_placement
    
    def generate_step_traj(self, leg: str, traj_type: str, delta_x: float, delta_y: float, delta_z: float=0.0, n: int=150) -> np.array: 
        """
        Generate a parabolic trajectory for the chosen leg.
        
        leg: Must be 'FR', 'FL', 'RR', 'RL'
        traj_type: Must be 'line', 'parabola'
        delta_x:  The distance to cover in the x axis.
        n: Number of points on the trajectory
        delta_z: The distance to cover in the z axis.
        line_slope: Slope of the staright line.

        """
        if leg not in {'FR', 'FL', 'RR', 'RL'}:
            raise ValueError("Leg choosing error. Leg must be one of: ('FR', 'FL', 'RR', 'RL')")
        elif traj_type not in {'line', 'parabola', 'trapezoid'}:
            raise ValueError("Trajectory type choosing error. Traj. type must be one of: ('line', 'parabola')")
        
        leg_pos = self.env.get_site_pos(leg+"_touch_sensor")
        quat = self.env.robot_data.qpos[3:7] * 1

        x_to = leg_pos[0] + delta_x
        y_to = leg_pos[1] + delta_y

        if traj_type == 'line':
            traj = self.step_line(leg_pos_0=leg_pos, n=n, dz=delta_z, quat=quat)
        elif traj_type == 'parabola':
            traj = self.parabola_trajectory(leg_pos_0=leg_pos, x1=x_to, z_max=leg_pos[2]+delta_z, n=n, y1=y_to)
        elif traj_type == 'trapezoid':
            traj = self.trapezoid_trajectory(leg_pos_0=leg_pos, x1=x_to, z_max=leg_pos[2]+delta_z, n=n, y1=y_to)
            if len(traj) < 2:
                self.no_traj = True
        return traj
    
    def get_deltas(self, leg_id, des_angles):
        z_des = 0.235
        self.pitch_mul = 1
        
        fr = self.env.get_site_pos("FR_touch_sensor")
        fl = self.env.get_site_pos("FL_touch_sensor")
        rr = self.env.get_site_pos("RR_touch_sensor")
        rl = self.env.get_site_pos("RL_touch_sensor")
        
        z_f = (fl[2]+fr[2])/2
        z_r = (rl[2]+rr[2])/2
        x_f = (fl[0]+fr[0])/2
        x_r = (rl[0]+rr[0])/2

        angles = des_angles
        if leg_id == "RR" or leg_id == "RL":
            if (z_f - z_r) < 0.0:
                x = 0.040/np.cos(angles[0])
                # print('x after cos mul: ', x)
            else:
                z_des = 0.255
                x = 0.040
        else: 
            if (z_f - z_r) > 0.01:
                z_des = 0.255
                x = 0.045
            else: x = 0.065

        if (z_f - z_r) < 0.01:
                x *= 2.5
                z_des = 0.255
                self.pitch_mul = 1.5
                # self.pitch_mul = 1.

        if (z_f - z_r) > 0.01:
                x *= 0.75
                z_des = 0.255
                self.pitch_mul = 0.5
                # self.pitch_mul = 1.

        y = 0.16029
        self.now_z_l = self.env.get_site_pos(leg_id+"_touch_sensor")[2]
        self.now_z_h = self.env.get_joint_anchor(leg_id+"_hip_joint")[2]
        z_height_divider = 3
        z = (self.now_z_h - self.now_z_l)/z_height_divider

        return x, y, z, z_des
    
    def get_final_z_pos(self, z_des):
        fr = self.env.get_site_pos("FR_touch_sensor")
        fl = self.env.get_site_pos("FL_touch_sensor")
        rr = self.env.get_site_pos("RR_touch_sensor")
        rl = self.env.get_site_pos("RL_touch_sensor")
        
        z_f = (fl[2]+fr[2])/2
        z_r = (rl[2]+rr[2])/2
        
        z_final = z_des - abs(z_f - z_r)

        return z_final

    def update_params(self, ):
        pass
    
    def parabola_trajectory(self, leg_pos_0, x1, y1, z_max, n) -> np.ndarray:
        n = n * 2
        x0, y0, z0 = leg_pos_0[0], leg_pos_0[1], leg_pos_0[2]
        quat = self.env.robot_data.qpos[3:7]
        com3 = self.env.robot_data.qpos[:3]
        
        x0_rot, y0_rot, z0_rot = quat_rot_mj(vec=leg_pos_0, q=quat, center_of_rotation=com3, inv=False)
        x1_rot, y1_rot, z1_rot = quat_rot_mj(vec=np.asarray([x1, y1, z0]), q=quat, center_of_rotation=com3, inv=False)
        
        # Calculate the vertex (h, k) of the parabola
        h = (x0 + x1) / 2
        h_rot = (x0_rot + x1_rot) / 2

        # Calculate the corresponding y values using the parabolic equation
        y_values_rot = np.linspace(y0_rot, y1_rot, n)
        y_values = np.linspace(y0, y1, n)
        # y_values = np.linspace(y0, y1, n)
        middle_y = y_values[(len(y_values)//2)]

        h1_rot, y_mid_rot, z_max_rot = quat_rot_mj(vec=np.asarray([h, middle_y, z_max]), q=quat, center_of_rotation=com3, inv=False)

        k = z_max
        k_rot = z_max_rot

        # Calculate the coefficient a of the parabolic equation
        a_rot = (z0_rot - k_rot) / ((x0_rot - h_rot) ** 2)
        
        # Generate n evenly spaced x values between x0 and x1
        x_end = x1_rot + 0.195
        x_values = np.linspace(x0_rot, x_end, n)

        z_values = a_rot * (x_values - h1_rot) ** 2 + k_rot
        # if np.diff(z_values)[0] < 0:
        #     # print('Wrong step trajectory values. Check if needed.')
        #     traj = np.asarray([[x0, y0, z0]])
        #     return traj
        
        zero_vec = com3
        traj = np.array([x_values, y_values_rot, z_values]).T
        traj_new = np.asarray([quat_rot_mj(point, quat, center_of_rotation=zero_vec, inv=True) for point in traj])

        return traj_new

    def parabola_trajectory_v_01(self, leg_pos_0, x1, y1, z_max, n) -> np.ndarray:
        n = n * 2
        x0, y0, z0 = leg_pos_0[0], leg_pos_0[1], leg_pos_0[2]
        
        # Calculate the vertex (h, k) of the parabola
        x_middle = (x0 + x1) / 2

        # Calculate the corresponding y values using the parabolic equation
        y_values = np.linspace(y0, y1, n)

        # Calculate the coefficient a of the parabolic equation
        a = (z0 - z_max) / ((x0 - x_middle) ** 2)
        
        # Generate n evenly spaced x values between x0 and x1
        x_end = x1 + 0.195
        x_values = np.linspace(x0, x_end, n)

        # Vertex form of parabola
        z_values = a * (x_values - x_middle) ** 2 + z_max
        traj = np.array([x_values, y_values, z_values]).T        
        return traj

    def parabola_trajectory_v_02(self, leg_pos_0, x1, y1, z_max, n) -> np.ndarray:
        n = n * 2
        x0, y0, z0 = leg_pos_0[0], leg_pos_0[1], leg_pos_0[2]
        quat = self.env.robot_data.qpos[3:7]
        com3 = self.env.robot_data.qpos[:3]
        


        x0_rot, y0_rot, z0_rot = quat_rot_mj(vec=leg_pos_0, q=quat, center_of_rotation=com3, inv=False)
        x1_rot, y1_rot, z1_rot = quat_rot_mj(vec=np.asarray([x1, y1, z0]), q=quat, center_of_rotation=com3, inv=False)
        
        # Calculate the vertex (h, k) of the parabola
        h = (x0 + x1) / 2
        h_rot = (x0_rot + x1_rot) / 2

        # Calculate the corresponding y values using the parabolic equation
        y_values_rot = np.linspace(y0_rot, y1_rot, n)
        y_values = np.linspace(y0, y1, n)
        # y_values = np.linspace(y0, y1, n)
        middle_y = y_values[(len(y_values)//2)]

        h1_rot, y_mid_rot, z_max_rot = quat_rot_mj(vec=np.asarray([h, middle_y, z_max]), q=quat, center_of_rotation=com3, inv=False)

        k = z_max
        k_rot = z_max_rot

        # Calculate the coefficient a of the parabolic equation
        a_rot = (z0_rot - k_rot) / ((x0_rot - h_rot) ** 2)
        
        # Generate n evenly spaced x values between x0 and x1
        x_end = x1_rot + 0.095
        x_values = np.linspace(x0_rot, x_end, n)

        z_values = a_rot * (x_values - h1_rot) ** 2 + k_rot
        # if np.diff(z_values)[0] < 0:
        #     # print('Wrong step trajectory values. Check if needed.')
        #     traj = np.asarray([[x0, y0, z0]])
        #     return traj
        
        zero_vec = com3
        traj = np.array([x_values, y_values_rot, z_values]).T
        traj_new = np.asarray([quat_rot_mj(point, quat, center_of_rotation=zero_vec, inv=True) for point in traj])

        return traj_new

    def line(self, x0, y0, x1, y1, m, n, z, z_end) -> np.ndarray:
        # Calculate y1 using the slope m
        # y1 = m * (x1 - x0) + y0
        
        # Generate n equally spaced x and y coordinates between x0, y0 and x1, y1
        x = np.linspace(x0, x1, n)
        y = np.linspace(y0, y1, n)

        z = np.linspace(z, z_end, n)

        # Combine x and y coordinates into an array of points
        points = np.column_stack((x, y, z))

        return points.transpose()
    
    def step_line(self, leg_pos_0, n, dz, quat) -> np.ndarray:
        
        body_0 = quat_rot_mj(vec=leg_pos_0, q=quat)
        z_end = body_0[2] + dz
        z = np.linspace(body_0[2], z_end, n)
        
        # ### Vectorizing:
        # points = np.asarray([quat_rot_mj(vec=np.asarray([body_0[0], body_0[1], z_val]), q=quat, inv=True) for z_val in z])
        body_0_repeated = np.tile(body_0[:2], (n, 1))
        points = np.hstack((body_0_repeated, z[:, np.newaxis]))

        # Combine x and y coordinates into an array of points
        # points = np.column_stack((x, y, z))

        # return points
        return np.apply_along_axis(quat_rot_mj, 1, points, q=quat, inv=True)

    def trapezoid_trajectory(self, leg_pos_0, x1, y1, z_max, n) -> np.ndarray:
        n = n//2
        z_max += 0.02
        x, y, z = leg_pos_0
        first_line_z = np.linspace(z, z_max, n).reshape(n, 1)
        first_line_x = np.zeros((n, 1)) + x
        first_line_y = np.zeros((n, 1)) + y
        first = np.stack([first_line_x, first_line_y, first_line_z], axis=1)[:, :, 0]

        second_line_x = np.linspace(x, x1, n).reshape(n, 1)
        second_line_y = np.linspace(y, y1, n).reshape(n, 1)
        second_line_z = np.zeros((n, 1)) + z_max
        second = np.stack([second_line_x, second_line_y, second_line_z], axis=1)[:, :, 0]

        third_line_z = np.linspace(z_max, z-0.2, n*2).reshape(n*2, 1)
        third_line_x = np.zeros((n*2, 1)) + x1
        third_line_y = np.zeros((n*2, 1)) + y1
        third = np.stack([third_line_x, third_line_y, third_line_z], axis=1)[:, :, 0]

        traj = np.row_stack([first, second, third])
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot3D(*traj, linewidth=2)
        # ax.set_xlim([-2.5, 2.5])
        # ax.set_ylim([-2.5, 2.5])
        # ax.set_zlim([-2.5, 2.5])

        # plt.show()
        return traj

    def circle_step(self, leg_pos_0, radius):
        rot_mat = self.env.get_body_rot(body_name='trunk')
        new_point = -rot_mat[:, 0] * radius + leg_pos_0
        new_vector = leg_pos_0.reshape(3, 1) + np.outer(self.points, new_point - leg_pos_0).T

        # x = radius * np.cos(two_pi) + new_point[0]
        # y = radius * np.sin(two_pi) + new_point[1]
        # z = np.zeros(n, ) + new_point[2]

        # arc = (rot_mat @ [x - new_point[0], y - new_point[1], z - new_point[2]]) + new_point.reshape(3, 1)
        # return np.stack([new_vector.T, arc.T], axis=0)
        self.after_circle = True
        return new_vector.T

# ### DRL Functions and classes:
def parse_args(parser):
    # fmt: off
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="MyGo1Env",
    # parser.add_argument("--gym-id", type=str, default="HalfCheetahBulletEnv-v0",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
    # parser.add_argument("--learning-rate", type=float, default=2e-5,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--save-interval", type=int, default=7,
        help="save model intervals")
    parser.add_argument("--num-trajectories", type=int, default=5,
        help="number of trajectories agent trained on")
    parser.add_argument("--num-points-paw", type=int, default=160,
        help="number of points in each simulation step of paw")
    parser.add_argument("--num-points-com", type=int, default=160,
        help="number of points in each simulation step of com")
    parser.add_argument("--actor-layer_size", type=int, default=512,
        help="number of neurons in actor layer")
    parser.add_argument("--critic-layer_size", type=int, default=512,
        help="number of neurons in critic layer")
    # parser.add_argument("--total-timesteps", type=int, default=2000000,
    parser.add_argument("--total-timesteps", type=int, default=80000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
    # parser.add_argument("--num-steps", type=int, default=1024,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=8,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.35,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=-0.005,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym_id
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -20, 20))
        # env.seed(seed)
        env.action_space.seed(seed)
        # env.observation_space.seed(seed)
        return env

    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, actor_layer_size, critic_layer_size):
        super(Agent, self).__init__()
        print('Single obs space shape: ', envs.single_observation_space.shape)
        self.checkpoint_file = 'ppo_model'

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), critic_layer_size)),
            nn.SiLU(),
            layer_init(nn.Linear(critic_layer_size, critic_layer_size)),
            nn.SiLU(),
            layer_init(nn.Linear(critic_layer_size, critic_layer_size)),
            nn.SiLU(),
            layer_init(nn.Linear(critic_layer_size, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), actor_layer_size*2)),
            nn.SiLU(),
            layer_init(nn.Linear(actor_layer_size*2, actor_layer_size)),
            nn.SiLU(),
            layer_init(nn.Linear(actor_layer_size, actor_layer_size)),
            nn.SiLU(),
            layer_init(nn.Linear(actor_layer_size, actor_layer_size)),
            nn.SiLU(),
            layer_init(nn.Linear(actor_layer_size, actor_layer_size//4)),
            nn.SiLU(),
            layer_init(nn.Linear(actor_layer_size//4, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        # self.actor_logstd = nn.Parameter(torch.full((1, np.prod(envs.single_action_space.shape)), 0.05))
        # self.actor_logstd = nn.Parameter(0.05 + 0.01 * torch.randn(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        try:
            probs = Normal(action_mean, action_std)
        except:
            print('Get action failed. Check settings.')
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def save_checkpoint(self, optimizer, epoch, loss, global_step, run_name, filename):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'global_step': global_step,
            'run_name': run_name
        }, filename)

    def load_checkpoint(self, optimizer, filename):
        checkpoint = torch.load(filename)
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # return checkpoint['epoch'], checkpoint['loss'], checkpoint['global_step']
        return checkpoint['epoch'], checkpoint['loss'], checkpoint['global_step'], checkpoint['run_name']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    profiler = LineProfiler()
    np.set_printoptions(precision=4)
    # matplotlib.use("Qt5agg")
    sb.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

    now = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
    xml_file = '/home/omri/Thesis/go1/xml/go1.xml'
    state_file = 'state_file.txt'
    env1 = Go1Env(xml=xml_file, env_num=0)
    my_ikine = IKine(env=env1)

    leg_id_row = {'FR': 7, 'FL': 10, 'RR': 13, 'RL': 16}
    leg_list = ['FR', 'RL', 'FL', 'RR']
    legs_pair = ['FL', 'RR', 'FR', 'RL']
    legs_infront = ['RR', 'FL', 'RL', 'FR']

    body_length_reducer = 1.
    traj_planner = Planner(env=env1, leg_list=leg_list, solver=my_ikine,
                        body_l=0.4*body_length_reducer, body_w=0.1)
    
    traj_maker = GenerateTrajectory((.5, -.5,), (2., 2.), 200)
    # traj_maker = GenerateTrajectory((.3, -.3,), (2., 2.), 200)
    sin_1 = traj_maker.sin_wave(A=1.2*5, omega=1.7)
    sin_2 = traj_maker.cosin_wave(A=0.8*5, omega=2.1)
    sin_3 = traj_maker.sin_wave(A=1.7*5, omega=.3)
    sin_4 = traj_maker.cosin_wave(A=0.5*5, omega=1.9)

    trajectory = sin_1 + sin_2 + sin_3 + sin_4
    trajectory = trajectory[:200, :200]
    # traj_fig = plt.figure()
    # plt.plot(trajectory)
    # plt.show(block=False)
    trajectories = traj_maker.generate_n_trajectories(args.num_trajectories)
    # plt.figure(666)
    # for traj in trajectories:
    #     plt.plot(traj[:, 0], traj[:, 1])
    # # plt.show(block=False)
    # plt.show(block=True)
    controller = PurePursuit(path=trajectories[0], lookahead_distance=0.3, 
                            wheelbase_length=0.3, max_steer_angle=0.10)

    p = np.array([500, 500, 200, 500, 500, 200, 500, 500, 200, 500, 500, 200])*1.5*1
    i = np.array([.01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01])
    d = np.array([.7, .8, .6, .7, .8, .6, .7, .8, .6, .7, .8, .6])*12*1.2*1
    my_pid = PIDController(P_gain=p, I_gain=i, D_gain=d, env=env1)

    test_and_not_run = True
    # plot_track = True
    plot_track = False
    test_and_not_run = False
    ### Tester env:
    if test_and_not_run:
        walker = Walker(env=env1, planner=traj_planner, controller=controller, trajectories2d=trajectories, 
                        leg_list=leg_list, leg_id_row=leg_id_row, legs_pair=legs_pair, block_render=False,
                        render_from_step=1, frames_to_skip_com=1, frames_to_skip_paw=1,
                        n_traj_for_com=args.num_points_com, n_traj_for_paw=args.num_points_paw, plot_tracking=plot_track, every_x_ep=1, PID=my_pid)

        des_yaw = -0.15
        completed = 0
        walker.reset()
        while True:
            delta = controller.compute_steering_angle(env1.robot_data.qpos[:2], walker.des_angles[-1])
            des_yaw += delta
            # print('Desired Yaw Angle: ', des_yaw, 'Delta Angle: ', delta)
            for idx, leg in enumerate(leg_list):
                move_com_trajectory = walker.move_com(leg, des_yaw, )
                lift_paw_trajectory = walker.lift_paw(leg, )
                parabolic_trajectory = walker.paw_step(leg, )
                # if parabolic_trajectory == -1: # Means that no parabolic trajectory could be found
                #     lift_paw_trajectory = walker.lift_paw(leg, put_leg_down=True)
                
            # env.write_state(state_file=state_file, 
            #                 state=env.info['State'], 
            #                 angles=env.info['Eul_Angles'])
            
            completed += 1
            print('Total Steps So Far: ', walker.tot_steps)

    else:
        first_env_block_render = True
        first_env_block_render = False
        plot_tracking = True
        plot_tracking = False
        walker_envs_values = create_walkers(n=24, render_first=first_env_block_render, plot_tracking=plot_tracking, xml_file=xml_file, 
                                            leg_list=leg_list, trajectories=trajectories, leg_id_row=leg_id_row, legs_pair=legs_pair, 
                                            body_length_reducer=body_length_reducer, args=args
                                            )

# TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


    class CustomCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(CustomCallback, self).__init__(verbose)
        
        def _on_step(self) -> bool:
            # Log custom metrics here
            # This example assumes your policy outputs log probabilities and values
            # Adjust according to your policy's output
            infos = self.locals['infos']
            for info in infos:
                if 'action_log_probs' in info and 'values' in info:
                    self.logger.record('custom/action_log_probs_mean', np.mean(info['action_log_probs']))
                    self.logger.record('custom/values_mean', np.mean(info['values']))
            return True

    # def make_env():
    #     return YourCustomEnv()  # Replace with your custom environment

    # check_env(walker_envs_values[0])

    # env setup
    # args.num_envs = len(walker_envs_values)
    args.num_envs = 24
    args.num_steps = 2048*args.num_envs // args.num_envs
    # args.num_steps = 500 // args.num_envs
    # args.num_steps = 4096 // args.num_envs
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    
    # envs = gym.vector.AsyncVectorEnv(
    #     # [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    #     [make_env(walker_envs_values[i], args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    # )
    envs = gym.vector.AsyncVectorEnv(
        # [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
        [make_env(walker_envs_values[i], args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    
    # # Number of parallel environments
    # n_envs = 12
    # env = make_vec_env(envs, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    # model = PPO("MlpPolicy", envs, verbose=1, tensorboard_log="./ppo_tensorboard/")
    # custom_callback = CustomCallback()

    # # Train the model
    # total_timesteps = 10000
    # model.learn(total_timesteps=total_timesteps, callback=[custom_callback])

    # # Save the model
    # model.save("ppo_custom_env")


    ### My Change:
    agent = Agent(envs, 
                  actor_layer_size=args.actor_layer_size,
                  critic_layer_size=args.critic_layer_size).to(device)
    model_str = str(agent)
    print(model_str)
    # parser.add_argument('--model-architecture', type=str, default=model_str, help="Model's architecture")
    args.model_arch = [model_str]
    args.model_reward_name = get_current_reward_func(Walker, 'get_reward')

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    global_step = 0

    continue_training = True
    continue_training = False

    # to_load = "C:\\Users\\Omri\\Desktop\\MSc\\Thesis\\ppo_trainings\\_MyGo1Env__Go1_KinematicWalker_Classes_v_0_03_Correct_ep_length__1__1696793572_checkpoint406.pth"
    # agent.load_checkpoint(optimizer=optimizer, filename=to_load)

    # def test(env, agent, test_eps):
    #     with torch.no_grad():
    #         for ep_count in range(test_eps):
    #             state = torch.Tensor(env.reset()[0]).to(device)
    #             done = False
    #             ep_reward = 0
    #             episode_step_count = 0

    #             while not done:
    #                 action, _, _, _ = agent.get_action_and_value(state)
    #                 next_state, reward, done, info, _ = env.step(action.cpu().numpy())
    #                 state = torch.Tensor(next_state).to(device)
    #                 ep_reward += reward
    #                 episode_step_count += 1

    #             print('Ep: {}, Ep Score: {}'.format(ep_count, ep_reward))

    # test(env=walker_envs_values[0], agent=agent, test_eps=10)

    if continue_training:
        checkpoint_file = 'C:\\Users\\Omri\\Desktop\\MSc\\Thesis\\Code\\Only_Code\\ppo_trainings\\_MyGo1Env__Go1_KinematicWalker_Classes_v_0_07_Train__1__1708187991_checkpoint77.pth'
        try:
            start_epoch, prev_loss, global_step, run_name = agent.load_checkpoint(optimizer, checkpoint_file)
            run_name = 'MyGo1Env__Go1_KinematicWalker_Classes_v_0_07_Train__1__1708187991'
            args.learning_rate = 2.98e-4
            print('\n\nModel Loaded, continue training.\n\n')
        except:
            print('\n\nERROR LOADING MODEL. RESTART.\n\n')
    else:
        start_epoch = 0

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs\\{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    # global_step = 0 # Defined above
    ep_num = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device) ### My Change - added [0] to take only obs and not info
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    ep_reward = 0.
    training_epochs = 0
    
    # agent.save_checkpoint(optimizer=optimizer, 
    #                           epoch=training_epochs, 
    #                           loss=0, 
    #                           global_step=0,
    #                           run_name=run_name,
    #                           filename='.\\ppo_trainings\\_'+run_name+'_checkpoint{}.pth'.format(training_epochs))
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)

                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, _, info = envs.step(action.cpu().numpy())
            ep_reward += reward[0]
            if done[0]:
                print('Episode reward:', ep_reward)
                ep_reward = 0
            # next_obs, reward, done, info, = envs.step(action.cpu().numpy())
            # print(reward)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            
            if "final_info" in info.keys():
                for idx, item in enumerate(info['final_info']):
                    log_step_check = 0
                    if item is None:
                        continue
                    try:
                        if item['episode']['l'] > 1:
                            time.sleep(0.05)
                            ep_num += 1
                            print(f"global_step={global_step+idx}, env_num={item['episode']['env_num']}, episodic_return={item['episode']['r']}, episodic_length={item['episode']['l']}, Checkpoint={item['episode']['Checkpoint']}")
                            if item['episode']['Checkpoint']:
                                writer.add_scalar("charts/Trajectory Completed - Checkpoint", item["episode"]["got_to_goal"], global_step+idx)
                                log_step_check = 1
                                writer.add_scalar("charts/Episodic Return Checkpoint", item["episode"]["r"], global_step+idx)
                                log_step_check = 2
                                writer.add_scalar("charts/Episode Length Checkpoint", item["episode"]["l"], global_step+idx)
                                log_step_check = 3
                            else:
                                writer.add_scalar("charts/Trajectory Completed - From Init. State", item["episode"]["got_to_goal"], global_step+idx)
                                log_step_check = 4
                                writer.add_scalar("charts/Episodic Return From Init. State", item["episode"]["r"], global_step+idx)
                                log_step_check = 5
                                writer.add_scalar("charts/Episode Length From Init. State", item["episode"]["l"], global_step+idx)
                            log_step_check = 6
                            writer.add_scalar("charts/Traj. Completed - Binary", item["episode"]["got_to_goal"], global_step+idx)
                            log_step_check = 7
                            writer.add_scalar("charts/Episodic Return", item["episode"]["r"], global_step+idx)
                            log_step_check = 8
                            writer.add_scalar("charts/Episodic Length", item["episode"]["l"], global_step+idx)
                            log_step_check = 9
                            writer.add_scalar("charts/Traj. Completed - Binary - By Ep.", item["episode"]["got_to_goal"], ep_num)
                            log_step_check = 10
                            writer.add_scalar("charts/Episodic Return - By Ep.", item["episode"]["r"], ep_num)
                            log_step_check = 11
                            writer.add_scalar("charts/Episodic Length - By Ep.", item["episode"]["l"], ep_num)
                            log_step_check = 12
                    except Exception as e:
                        print(f'Logging error at step {global_step+idx}, check the logger. Exception: {e}, at log num: {log_step_check}')
                    writer.flush()
        
        # checkpoint value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        training_epochs += 1
        if training_epochs % args.save_interval == 0:
            agent.save_checkpoint(optimizer=optimizer, 
                                epoch=training_epochs, 
                                loss=loss,
                                global_step=global_step, 
                                run_name=run_name,
                                filename='./ppo_trainings/_'+run_name+'_checkpoint{}.pth'.format(training_epochs+start_epoch))
        
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/Learning rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/Value loss", v_loss.item(), global_step)
        writer.add_scalar("losses/Policy loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/Entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/Old approx kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/Approx kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/Clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/Explained variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
