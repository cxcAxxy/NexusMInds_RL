from typing import Any, Dict
import torch
from ..core import Task
from .utils import distance  # distance函数也需支持torch张量

class Reach(Task):
    def __init__(
            self,
            sim,
            cfg
    ) -> None:
        super().__init__(sim)

        self.sim = sim
        self.device = cfg.device  # 新增 device 属性
        self.num_envs= cfg.num_envs
        self.goal = torch.tensor((0.2 , 0.1 , 0),device=self.device)

        self.reward_type = cfg.reward_type
        self.distance_threshold = torch.tensor(cfg.distance_threshold, dtype=torch.float32, device=self.device)
        self.goal_range_low = torch.tensor([-cfg.goal_range / 2, -cfg.goal_range / 2, 0.0],
                                           dtype=torch.float32, device=self.device)
        self.goal_range_high = torch.tensor([cfg.goal_range / 2, cfg.goal_range / 2, cfg.goal_range],
                                            dtype=torch.float32, device=self.device)

        self._create_scene()

    def _create_scene(self) -> None:
        # self.sim.create_plane(z_offset=-0.4)
        #self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=torch.zeros(3, dtype=torch.float32, device=self.device),
            orn=torch.tensor([0,0,0,1],device=self.device)
        )

    def get_obs(self):
        return None

    def get_achieved_goal(self) -> torch.Tensor:
        ee_position = self.sim.get_ee_position()
        return ee_position

    # 这个地方的旋转就使用标准的 (0,0,0,1)即可，然后名字也是要一定对应的
    def reset_idx(self,idx : list) -> None:
        self.sim.set_actor_pose("target",idx, self.goal,torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=self.device))


    def reset(self):
        self.reset_idx(list(range(self.num_envs)))

    # def _sample_goal_rand(self) -> torch.Tensor:
    #     """
    #     Randomize goals for multiple environments without using torch.Generator.
    #     """
    #     goal = (self.goal_range_high - self.goal_range_low) * \
    #            torch.rand((self.num_envs, 3), device=self.device) + self.goal_range_low
    #     return goal

    def is_success(self, achieved_goal: torch.Tensor) -> torch.Tensor:
        d = distance(achieved_goal, self.goal)  # distance函数需支持torch张量
        return d < self.distance_threshold

    def compute_reward(self, achieved_goal: torch.Tensor) -> torch.Tensor:
        d = distance(achieved_goal,self.goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).float()
        else:
            return -d
