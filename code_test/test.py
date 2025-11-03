from env.Robot.gym_env.sim import Gym
import numpy as np

from env.core import Robot

asset_root="/home/cxc/Desktop/NexusMind_rl/env/assets"

urdf_file="urdf/franka_description/robots/franka_panda.urdf"

base_pos=np.array([0,0,0])
base_orn=np.array([0,0,0,1])

from isaacgym import gymapi, gymutil

import torch
def test():
    args = gymutil.parse_arguments(
        description="test Gym Simulation",
        custom_parameters=[
            {"name": "--num_envs", "type": int, "default": 6, "help": "The nums of env"},
            {"name": "--control_type", "type": str, "default": "effort", "help": "the control model" },
            {"name": "--headless", "type": bool, "default": False, "help": "Run simulation without viewer"},
            {"name": "--use_gpu", "type": bool, "default": True, "help": "Use GPU for physics"},
            {"name": "--use_gpu_pipeline", "type": bool, "default": True, "help": "Use GPU pipeline"}

        ]
    )
    print(args)
    gym_test=Gym(args)
    gym_test.pre_simulate(asset_root,urdf_file,base_pos,base_orn)

    init_ee_pos=gym_test.get_ee_position()
    init_ee_orn=gym_test.get_ee_orientation()

    print("init_ee_pos")
    print(init_ee_pos)
    print("init_ee_orn")
    print(init_ee_orn)

    offset = torch.tensor([-0.4, -0.2, 0.2], device=init_ee_pos.device)
    target_pos = init_ee_pos + offset

    print("target_pos")
    print(target_pos)
    target_orn=init_ee_orn

    while True:
        u=gym_test.ee_pos_to_torque(target_pos,target_orn)
        gym_test.step(u)


if __name__ == "__main__":
    test()