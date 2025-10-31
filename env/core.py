from abc import ABC, abstractmethod

# 这个是pytho内置模块abc，用来创建 抽象类和抽象方法,abstractmethod,用来标记子类必须实现的方法，子类如果没有实现这些方法，也不能实例化
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from mpmath.libmp import dps_to_prec


# 这个就是抽象模板类几何



class Robot(ABC):
    """Base class for robot env.

    Args:
        sim (isaac gym or MuJoCo): Simulation instance.
        body_name (str): The name of the robot within the simulation.
        file_name (str): Path of the urdf file.
        base_position (np.ndarray): Position of the base of the robot as (x, y, z).
    """

    def __init__(self):
        pass

    # 将神经网络输出 转换成物理仿真引擎能理解的控制命令
    @abstractmethod
    def set_action(self, action: np.ndarray) -> None:
        """Set the action. Must be called just before sim.step().

        Args:
            action (np.ndarray): The action.
        """

    # 用于从仿真环境中提出机器人当前的观测信息。
    @abstractmethod
    def get_obs(self) -> np.ndarray:
        """Return the observation associated to the robot.

        Returns:
            np.ndarray: The observation.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the robot and return the observation."""


class Task(ABC):
    """Base class for tasks.
    Args:
        sim (issac gym or MuJuCo): Simulation instance.
    """

    def __init__(self, sim) -> None:
        self.sim = sim
        self.goal = None


    # 主要是 set a new goal,看任务需要重置吗，还是说一样。
    @abstractmethod
    def reset(self) -> None:
        """Reset the task: sample a new goal."""

    # return the observation associated to the task,这个后续比较有用，用于扩展。
    @abstractmethod
    def get_obs(self) -> np.ndarray:
        """Return the observation associated to the task."""


    #  这个achieved_goal理解为 当前机器人的已经达到的目标状态。
    @abstractmethod
    def get_achieved_goal(self) -> np.ndarray:
        """Return the achieved goal."""

    def get_goal(self) -> np.ndarray:
        """Return the current goal."""
        if self.goal is None:
            raise RuntimeError("No goal yet, call reset() first")
        else:
            return self.goal.copy()

    @abstractmethod
    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        """Returns whether the achieved goal match the desired goal."""

    @abstractmethod
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        """Compute reward associated to the achieved and the desired goal."""


class RobotTaskEnv():
    """Robotic task goal env, as the junction of a task and a robot.

    Args:
        robot (Robot): The robot.
        task (Task): The task.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Roll of the camera. Defaults to 0.
    """

    def __init__(
        self,
        robot: Robot,
        task: Task,
        cfg

    ) -> None:

        assert robot.sim == task.sim, "The robot and the task must belong to the same simulation."
        self.sim = robot.sim
        # self.render_mode = self.sim.render_mode
        # self.metadata["render_fps"] = 1 / self.sim.dt
        self.robot = robot
        self.task = task
        self.device=cfg.device


        self.num_envs=cfg.num_envs


        observation, _ = self.reset()  # required for init; seed can be changed later
        # 后面这个地方要改为,后面这个地方前面是环境的信息。
        self.num_obs = observation["observation"].shape
        self.num_privileged_obs=None    # 后续更新

        self.num_achieved_goal = observation["achieved_goal"].shape
        self.num_desired_goal = observation["desired_goal"].shape
        self.num_actions=self.robot.action_dim
        self.max_episode_length=cfg.max_episode_length

        # allocate buffers
        self.obs_buf=torch.zeros(self.num_envs,self.num_obs,device=self.device,dtype=torch.float)
        self.achieved_goal_buf=torch.zeros(self.num_envs,self.num_achieved_goal,device=self.device,dtype=torch.float)
        self.desired_goal_buf=torch.zeros(self.num_envs,self.num_desired_goal,device=self.device,dtype=torch.float)


        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device,dtype=torch.float)
        else:
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs

        self.extras = {}
        self.compute_reward = self.task.compute_reward


    def get_obs(self):
        return self.obs_buf

    def get_achieved_goal_obs(self):
        return self.achieved_goal_buf

    def get_desired_goal_obs(self):
        return self.desired_goal_buf

    #重置特定的环境
    def reset_idx(self,env_ids):
        if len(env_ids) == 0:
            return

        self.robot.reset_ids(env_ids)
        self.task.reset_ids(env_ids)

        #重置buffer的变量
        self.rew_buf[env_ids]=0.
        self.episode_length_buf[env_ids]=0.
        self.time_out_buf[env_ids]=0.
        self.reset_buf[env_ids]=1.

        # fill extras
        # self.extras["episode"] = {}
        # for key in self.episode_sums.keys():
        #     self.extras["episode"]['rew_' + key] = torch.mean(
        #         self.episode_sums[key][env_ids]) / self.max_episode_length_s
        #     self.episode_sums[key][env_ids] = 0.
        # if self.cfg.commands.curriculum:
        #     self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # # send timeout info to the algorithm
        # if self.cfg.env.send_timeouts:
        #     self.extras["time_outs"] = self.time_out_buf

    def reset(self):

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs


    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.robot.set_action(action)

        self.sim.step()

        observation = self.get_obs()
        # An episode is terminated iff the agent has reached the target
        terminated = bool(self.task.is_success(observation["achieved_goal"], self.task.get_goal()))
        truncated = False
        info = {"is_success": terminated}
        reward = float(self.task.compute_reward(observation["achieved_goal"], self.task.get_goal(), info))
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.sim.close()

    # def render(self) -> Optional[np.ndarray]:
    #     """Render.
    #
    #     If render mode is "rgb_array", return an RGB array of the scene. Else, do nothing and return None.
    #
    #     Returns:
    #         RGB np.ndarray or None: An RGB array if mode is 'rgb_array', else None.
    #     """
    #     return self.sim.render(
    #         width=self.render_width,
    #         height=self.render_height,
    #         target_position=self.render_target_position,
    #         distance=self.render_distance,
    #         yaw=self.render_yaw,
    #         pitch=self.render_pitch,
    #         roll=self.render_roll,
    #     )
