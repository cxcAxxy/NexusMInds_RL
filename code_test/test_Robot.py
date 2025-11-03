
from isaacgym import gymapi, gymutil
from env.Robot.gym_env.sim.pygym import Gym

from env.Robot.gym_env.instance.franka import Franka
from configs.Robot_config import Config
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

    action=torch.tensor([[1,1,1],[0.5,0.5,0.5],[0.8,0.6,0.4]],device="cuda:0")

    for i in range (30):
        robot.step(action)



if __name__ == "__main__":
    test()