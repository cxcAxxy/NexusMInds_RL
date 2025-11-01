import numpy as np
from sympy.physics.units import action

from ....core import Robot
from ..sim import Gym

class Franka(Robot):
    def __init__(self,sim:Gym ,cfg):
        # 那么这个地方按照sim，就是以文档里面的 官方文档的prepare_sim为界限
        self.num_actions=cfg.num_actions
        self.num_obs=cfg.num_obs
        self.num_envs=cfg.num_envs
        self.sim=sim
        self.cfg=cfg
        #准备资产，创建环境，为后续的控制做好准备
        self.sim.pre_simulate(cfg.asset,cfg.robot_files,cfg.base_poses,cfg.base_ornes,cfg.num_envs)
        super.__init__()

    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        #action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.cfg.control_type == "ee":
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
        else:
            arm_joint_ctrl = action[:7]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

        if self.cfg.block_gripper:
            target_fingers_width = 0
        else:
            fingers_ctrl = action[-1] * 0.2  # limit maximum change in position
            fingers_width = self.sim.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl

        target_angles = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
        return target_angles


    def get_obs(self) -> np.ndarray:
        # end-effector position and velocity
        ee_position = np.array(self.sim.get_ee_position())
        ee_velocity = np.array(self.sim.get_ee_velocity())
        # fingers opening
        if not self.cfg.block_gripper:
            fingers_width = self.sim.get_fingers_width()
            observation = np.concatenate((ee_position, ee_velocity, [fingers_width]))
        else:
            observation = np.concatenate((ee_position, ee_velocity))
        return observation

    def reset(self) -> None:
        """Reset the robot and return the observation."""
        self.sim.set_joint_neutral()


    #后面是根据机器的模型，自己定义的一些函数，服务于set_action,get_obs。
    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        ee_displacement = ee_displacement[:3] * 0.05  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.sim.get_ee_position()
        target_ee_position = ee_position + ee_displacement


        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles

        target_arm_angles = self.sim.inverse_kinematics(
            link=self.sim.ee_link, position=target_ee_position, orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        target_arm_angles = target_arm_angles[:7]  # remove fingers angles
        return target_arm_angles


    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 7 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in position
        # get the current position and the target position
        current_arm_joint_angles = np.array([self.sim.get_joint_angle(joint=i) for i in range(7)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

