import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rsl_rl.modules import DDPGActorCritic
class ReplayBuffer:
    def __init__(self, max_size, num_envs, obs_dim, act_dim, device):
        self.max_size = max_size
        self.num_envs = num_envs
        self.size = 0
        self.device = device
        self.ptr = 0
        self.full = False

        # 为每个环境存储标量奖励和done标志
        self.obs_buf = torch.zeros((self.max_size, num_envs, obs_dim), dtype=torch.float32, device=self.device)
        self.next_obs_buf = torch.zeros((self.max_size, num_envs, obs_dim), dtype=torch.float32, device=self.device)
        self.act_buf = torch.zeros((self.max_size, num_envs, act_dim), dtype=torch.float32, device=self.device)
        self.rew_buf = torch.zeros((self.max_size, num_envs), dtype=torch.float32, device=self.device)  # 标量奖励
        self.done_buf = torch.zeros((self.max_size, num_envs), dtype=torch.bool, device=self.device)    # 布尔done
        
    def add(self, obs, act, rew, done, next_obs):

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size):
        # 随机索引，torch 版本
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)

        # 取出对应的 transitions
        obs = self.obs_buf[idx]          # [batch_size, num_envs, obs_dim]
        act = self.act_buf[idx]          # [batch_size, num_envs, act_dim]
        rew = self.rew_buf[idx]
        next_obs = self.next_obs_buf[idx]# [batch_size, num_envs, obs_dim]
        done = self.done_buf[idx]     

        # 如果希望展开 env 维度成标准训练 batch: [batch_size * num_envs, dim]
        obs = obs.reshape(-1, obs.shape[-1])
        act = act.reshape(-1, act.shape[-1])
        rew = rew.reshape(-1, 1)         # 展开后添加维度 [batch_size * num_envs, 1]
        next_obs = next_obs.reshape(-1, next_obs.shape[-1])
        done = done.reshape(-1, 1).float()  # 转换为float并添加维度 [batch_size * num_envs, 1]

        return obs, act, rew, next_obs, done

    def __len__(self):
        return self.size if self.full else self.ptr


# ================= DDPG/TD3-style Algorithm =================
class DDPG:
    """Off-policy TD3/DDPG-style algorithm using DDPGActorCritic"""
    actor_critic: DDPGActorCritic
    def __init__(self, actor_critic, device='cpu', gamma=0.99, tau=0.005, batch_size=256, lr_actor=1e-3,lr_critic=1e-3):
        self.ac = actor_critic
        self.actor_critic = actor_critic 
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.replay_buffer = None

        # Optimizers
        self.actor_opt = torch.optim.Adam(self.ac.actor.parameters(), lr=lr_actor)
        self.critic_opt = torch.optim.Adam([p for q in self.ac.q_networks for p in q.parameters()], lr=lr_critic)

    def init_storage(self, buffer_size, num_envs, obs_shape, act_shape):
        self.replay_buffer = ReplayBuffer(int(buffer_size), num_envs, obs_shape[0], act_shape[0], self.device)

    def store_transition(self, obs, act, rew, done, next_obs):
        self.replay_buffer.add(obs, act, rew, done, next_obs)

    def update(self):
        # 并行环境下可用样本总数
        total_samples = len(self.replay_buffer)
        if total_samples < self.batch_size:
            return None, None, None

        obs, act, rew, next_obs, done = self.replay_buffer.sample(self.batch_size)

        # Critic update
        with torch.no_grad():
            next_act = self.ac.actor_target(next_obs)
            target_q_list = [q(torch.cat([next_obs, next_act], dim=-1)) for q in self.ac.q_target]
            target_q = torch.min(torch.cat(target_q_list, dim=1), dim=1, keepdim=True)[0]
            target = rew + self.gamma * (1 - done) * target_q

        q_vals = self.ac.q_values(obs, act)
        critic_loss = sum(F.mse_loss(q_val, target) for q_val in q_vals) / len(q_vals)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update
        actor_loss = -self.ac.q1(obs, self.ac.actor(obs)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft update
        self.ac.soft_update(self.tau)

        mean_noise_std = getattr(self.ac, "noise_std", 0.0)
        return critic_loss.item(), actor_loss.item(), mean_noise_std

