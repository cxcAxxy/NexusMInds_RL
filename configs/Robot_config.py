class Config:
    num_actions = 3           # 7 个关节 + 1 手指
    num_obs = 4
    num_envs = 3
    control_type = "ee"
    control_type_sim = "effort"
    block_gripper = False
    asset = "/home/cxc/Desktop/NexusMind_rl/env/assets"
    robot_files = "urdf/franka_description/robots/franka_panda.urdf"
    base_pose = [0,0,0]  # 每个环境的机器人位置
    base_orn = [0,0,0,1] # 每个环境的机器人姿态
    ee_link = "panda_hand"
    headless = "False"
    control_decimation = 4
    action_low = -1
    action_high = 1

