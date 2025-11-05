from .utils import distance
from typing import Any, Dict
import torch
from ..core import Task

class Reach(Task):
    def __init__(self, sim, cfg) -> None:
        super().__init__(sim)
        self.sim = sim
        self.reward_type = cfg.reward_type
        self.distance_threshold = cfg.distance_threshold
        self.device = cfg.device
        self.num_envs = cfg.num_envs

        # 获取末端执行器位置的函数
        self.get_ee_position = sim.get_ee_position()

        # 目标范围
        self.goal_range_low = torch.tensor([-cfg.goal_range / 2, -cfg.goal_range / 2, 0.0],
                                           dtype=torch.float32, device=self.device)
        self.goal_range_high = torch.tensor([cfg.goal_range / 2, cfg.goal_range / 2, cfg.goal_range],
                                            dtype=torch.float32, device=self.device)

        # 初始化目标缓存 (num_envs, 3)
        self.goal = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

    def get_obs(self) -> torch.Tensor:
        """返回任务观测，可自行扩展"""
        return torch.empty((self.num_envs, 0), device=self.device)

    def get_achieved_goal(self) -> torch.Tensor:
        """获取所有环境的末端执行器位置 (num_envs, 3)"""
        ee_positions = self.sim.get_ee_position()
        return ee_positions


    def reset_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        """只为指定环境重置目标"""
        # goals = self._sample_goals(len(env_ids))
        # self.goal[env_ids] = goals

        # 固定目标位姿
        self.goal[env_ids] = torch.tensor([0.5, 0.3, 0.25], device=self.device).unsqueeze(0).repeat(len(env_ids), 1)
        # 通过设置的 goal 和 没有旋转的 orn
        pos = torch.tensor([0.3, 0.2, 0.1], device=self.device)  # 固定位置
        orn = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)  # 固定姿态
        self.sim.set_actor_pose("target", pos, orn,env_ids)


    def _sample_goals_rand(self, num_envs: int) -> torch.Tensor:
        """为若干环境随机生成目标 (num_envs, 3)"""
        rand = torch.rand((num_envs, 3), device=self.device)
        goals = self.goal_range_low + (self.goal_range_high - self.goal_range_low) * rand
        return goals

    def is_success(self, achieved_goal: torch.Tensor, desired_goal: torch.Tensor) -> torch.Tensor:
        """判断是否成功 (num_envs,)"""
        d = torch.norm(achieved_goal - desired_goal, dim=-1)
        return d < self.distance_threshold

    def compute_reward(self, achieved_goal: torch.Tensor, desired_goal: torch.Tensor) -> torch.Tensor:
        """计算奖励 (num_envs,)"""
        d = torch.norm(achieved_goal - desired_goal, dim=-1)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).float()
        else:
            return -d

