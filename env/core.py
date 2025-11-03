from abc import ABC, abstractmethod

# 这个是pytho内置模块abc，用来创建 抽象类和抽象方法,abstractmethod,用来标记子类必须实现的方法，子类如果没有实现这些方法，也不能实例化
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch



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


        observation,privileged_obs,achieved_goal,desired_goal,_,_,_ = self.reset()  # required for init; seed can be changed later
        # 后面这个地方要改为,后面这个地方前面是环境的信息。
        self.num_obs = observation["observation"].shape[1]
        self.num_privileged_obs=None    # 后续更新

        self.num_achieved_goal = observation["achieved_goal"].shape[1]
        self.num_desired_goal = observation["desired_goal"].shape[1]


        self.num_actions=self.robot.num_actions
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

        self.compute_reward_task=task.compute_reward
        self.extras = {}


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
        self.reset_buf[env_ids]=0.

        #fill extras
        self.extras["episode"] = {}

        self.extras["episode"]["goal_reward"]=torch.mean(self.rew_buf[env_ids])/self.cfg.max_episode_length_s
        # send timeout info to the algorithm
        self.extras["time_outs"]=self.time_out_buf

        #后续补充这些的
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
        #重置的另一种写法，按照初始的状态来,还有一点就是，应该还有各种的 achieved_goal,还有对应的goal
        obs, privileged_obs,achieved_goal,desired_goal, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))

        return obs, privileged_obs,achieved_goal,desired_goal,_,_,_


    def step(self, action: np.ndarray):
        #修改为 多环境的，dones,还有extras，对应就是内部的信息。
        action_sim=self.robot.set_action(action)

        # 这个地方设置 control.decimation。
        for _ in range(self.cfg.control.decimation):
            self.sim.step(action_sim)         # 这个地方一定要refesh，就是要更新数值，后面读取的一定是更新之后的。

        # 更新的问题！！！！，这个更新放到仿真环境里面，就是robot的接口一定要是完全合适的。
        self.post_physics_step()

        return self.obs_buf, self.privileged_obs_buf, self.achieved_goal_buf,self.desired_goal_buf,self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):

        #更新buf的数值。
        self.episode_length_buf += 1
        self.check_termination()

        # 顺序上的问题,注意一下
        self.update_observations()
        self.compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

    # 更新对应buffer数值。

    def check_termination(self):
        """ Check if environments need to be reset
        """
        #self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        #self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        # 这个地方的仿真接口，就是 reset_buf的判断条件.

        #这个地方也不进行一个更新，判断条件后面再说，根据任务后续设定

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        self.rew_buf=self.compute_reward_task(self.achieved_goal_buf,self.desired_goal_buf)

        # 目前奖励只有一个，后面可是使用,就是记录每一个的
        #self.episode_sums[name] += rew
        # 设置 是否只有正确奖励A
        # if self.cfg.rewards.only_positive_rewards:
        #     self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping


        # 维护各个奖励的reward_scalse这个具体的操作还有后续还要处理，然后termination感觉重复计算了
        # if "termination" in self.reward_scales:
        #     rew = self._reward_termination() * self.reward_scales["termination"]
        #     self.rew_buf += rew
        #     self.episode_sums["termination"] += rew

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf


    def update_observations(self):
        # 更新对应
        self.obs_buf=self.robot.get_obs()
        self.desired_goal_buf=self.task.get_goal()
        self.achieved_goal_buf=self.task.get_achieved_goal()

    def close(self) -> None:
        self.sim.close()
