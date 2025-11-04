import time

from env.Robot.gym_env.sim.pygym import Gym

from env.Task.Reach import Reach

from env.Robot.gym_env.instance.franka import Franka

from configs.Robot_config import Config
from configs.Task_config import ReachConfig

from isaacgym import gymapi, gymutil

import torch

def test():
    args = gymutil.parse_arguments(
        description="test Gym Simulation",
        custom_parameters=[
            {"name": "--use_gpu", "type": bool, "default": True, "help": "Use GPU for physics"},
            {"name": "--use_gpu_pipeline", "type": bool, "default": True, "help": "Use GPU pipeline"},
            {"name": "--headless", "type": bool, "default": False, "help": "Run simulation without viewer"},
        ]
    )
    cfg=Config()
    _Gym= Gym(args)
    robot=Franka(_Gym,cfg)
    task = Reach(_Gym,ReachConfig)


    action=torch.tensor([[1,1,1],[0.5,0.5,0.5],[0.8,0.6,0.4],[1,1,1]],device="cuda:0")

    for i in range (30):

        reward = task.compute_reward(task.sim.get_ee_position())
        print(reward)

        print(task.is_success(task.sim.get_ee_position()))
        robot.step(action)

    time.sleep(5)



if __name__ == "__main__":
    test()