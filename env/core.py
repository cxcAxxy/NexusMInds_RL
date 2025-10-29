from abc import ABC, abstractmethod

# 这个是pytho内置模块abc，用来创建 抽象类和抽象方法,abstractmethod,用来标记子类必须实现的方法，子类如果没有实现这些方法，也不能实例化
from typing import Any, Dict, Optional, Tuple
import numpy as np

# 这个就是抽象模板类几何



class Robot(ABC):
    """Base class for robot env.

    Args:
        sim (isaac gym or MuJoCo): Simulation instance.
        body_name (str): The name of the robot within the simulation.
        file_name (str): Path of the urdf file.
        base_position (np.ndarray): Position of the base of the robot as (x, y, z).
    """

    def __init__(
        self,
        sim,
        body_name: str,
        file_name: str,
        base_position: np.ndarray,
        action_dim: int,
        joint_indices: np.ndarray,
        joint_forces: np.ndarray,
    ) -> None:
        self.sim = sim
        self.body_name = body_name
        with self.sim.no_rendering():
            self._load_robot(file_name, base_position)
            self._callback_afterload()
        self.action_dim = action_dim
        self.joint_indices = joint_indices
        self.joint_forces = joint_forces

    def _load_robot(self, file_name: str, base_position: np.ndarray) -> None:
        """Load the robot.

        Args:
            file_name (str): The URDF file name of the robot.
            base_position (np.ndarray): The position of the robot, as (x, y, z).
        """
        self.sim.loadURDF(
            body_name=self.body_name,
            fileName=file_name,
            basePosition=base_position,
            useFixedBase=True,
        )

    def _callback_afterload(self) -> None:
        """Called after robot loading."""
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

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        robot: Robot,
        task: Task,
        # render_width: int = 720,
        # render_height: int = 480,
        # render_target_position: Optional[np.ndarray] = None,
        # render_distance: float = 1.4,
        # render_yaw: float = 45,
        # render_pitch: float = -30,
        # render_roll: float = 0,
    ) -> None:

        # 渲染逻辑，看到底应该如何渲染。
        assert robot.sim == task.sim, "The robot and the task must belong to the same simulation."
        self.sim = robot.sim
        # self.render_mode = self.sim.render_mode
        # self.metadata["render_fps"] = 1 / self.sim.dt
        self.robot = robot
        self.task = task


        observation, _ = self.reset()  # required for init; seed can be changed later
        observation_shape = observation["observation"].shape
        achieved_goal_shape = observation["achieved_goal"].shape
        desired_goal_shape = observation["desired_goal"].shape
        self.observation_space = dict(
            observation=np.random.uniform(-10.0, 10.0, size=observation_shape).astype(np.float32),
            desired_goal=np.random.uniform(-10.0, 10.0, size=desired_goal_shape).astype(np.float32),
            achieved_goal=np.random.uniform(-10.0, 10.0, size=achieved_goal_shape).astype(np.float32),
        )
        self.action_dim = self.robot.action_dim
        self.compute_reward = self.task.compute_reward
        self._saved_goal = dict()  # For state saving and restoring

        # self.render_width = render_width
        # self.render_height = render_height
        # self.render_target_position = (
        #     render_target_position if render_target_position is not None else np.array([0.0, 0.0, 0.0])
        # )
        # self.render_distance = render_distance
        # self.render_yaw = render_yaw
        # self.render_pitch = render_pitch
        # self.render_roll = render_roll
        # with self.sim.no_rendering():
        #     self.sim.place_visualizer(
        #         target_position=self.render_target_position,
        #         distance=self.render_distance,
        #         yaw=self.render_yaw,
        #         pitch=self.render_pitch,
        #     )

    def _get_obs(self) -> Dict[str, np.ndarray]:
        robot_obs = self.robot.get_obs().astype(np.float32)  # robot state
        task_obs = self.task.get_obs().astype(np.float32)  # object position, velocity, etc...
        observation = np.concatenate([robot_obs, task_obs])
        achieved_goal = self.task.get_achieved_goal().astype(np.float32)
        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": self.task.get_goal().astype(np.float32),
        }


    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:

        # self.task.np_random = self.np_random
        # with self.sim.no_rendering():
        self.robot.reset()
        self.task.reset()

        observation = self._get_obs()
        info = {"is_success": self.task.is_success(observation["achieved_goal"], self.task.get_goal())}
        return observation, info


    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.robot.set_action(action)

        self.sim.step()

        observation = self._get_obs()
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
