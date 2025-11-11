# config/franka_reach_cfg.py
import numpy as np
from isaacgym import gymutil
from .base_config import BaseConfig
import torch

args = gymutil.parse_arguments(
        description="test Gym Simulation",
        custom_parameters=[
            {"name": "--use_gpu", "type": bool, "default": True, "help": "Use GPU for physics"},
            {"name": "--use_gpu_pipeline", "type": bool, "default": True, "help": "Use GPU pipeline"},
            # {"name": "--headless", "type": bool, "default": True, "help": "Run simulation without viewer"},
             {"name": "--headless", "type": bool, "default": False, "help": "Run simulation without viewer"},
        ]
    )

class GymCfg:
    """仿真器配置"""
    def __init__(self, args=None):
        # 默认值
        self.headless = False
        self.use_gpu = True
        self.use_gpu_pipeline = True

        # 如果传了 args，就覆盖默认值
        if args is not None:
            for key, value in vars(args).items():
                setattr(self, key, value)



class RobotCfg:
    """机械臂配置"""
    def __init__(self):
        # 控制相关参数
        self.control_type = "ee"      
        self.block_gripper = True 
        self.num_actions = 3            
        self.num_obs = 6
        self.num_envs = 3 # 修改为与其他配置一致
        self.control_type_sim = "effort"             

        # 模型路径与姿态 - 修复资产路径
        self.asset = "/home/ymy/space_rl/NexusMInds_RL/env/assets"
        self.robot_files = "urdf/franka_description/robots/franka_panda.urdf"
        # 每个机器人的初始位置是一样的吗
        self.base_pose = [0,0,0]  # 每个环境的机器人位置
        self.base_orn = [0,0,0,1] # 每个环境的机器人姿态

        self.ee_link = "panda_hand"
        self.headless = "False"
        self.control_decimation = 6
        self.action_low = -1
        self.action_high = 1



class TaskCfg:
    """Franka Reach 任务配置"""
    def __init__(self):
        self.name = "Reach"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_envs = 3

        self.reward_type = "dense"
        self.distance_threshold = 0.05

        self.goal_range = 1
        self.get_ee_position = None

        # 调整轴向权重配置 [x, y, z] - 降低z轴权重，让训练更平衡
        self.axis_weights = [1.0, 1.0, 1.5]  


class AllCfg:
    """环境总体配置"""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_envs = 3
        self.num_achieved_goal = 3
        self.num_desired_goal = 3
        self.max_episode_length = 200
        self.max_episode_length_s = 4.0  # 秒数形式（用于日志统计）
        self.decimation = 4  
        self.control_type_sim = "effort"  


class FrankaReachCfg:
    """总配置类"""
    def __init__(self):
        self.gymcfg = GymCfg(args)
        self.robotcfg = RobotCfg()
        self.taskcfg = TaskCfg()
        self.all = AllCfg()




