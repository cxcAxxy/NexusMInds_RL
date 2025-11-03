class ReachConfig:
    # 环境相关
    num_envs = 4
    device="cuda:0"
    seed = 42



    # 机器人相关
    get_ee_position = None  # 需要在创建任务时传入函数

    # 奖励和成功判定
    reward_type = "dense"         # "sparse" 或 "dense"
    distance_threshold = 0.05     # 成功距离阈值

    # 目标采样范围
    goal_range = 0.5
    goal_range_low = [-goal_range / 2, -goal_range / 2, 0]
    goal_range_high = [goal_range / 2, goal_range / 2, goal_range]

    # 其他参数
    action_low = -1.0
    action_high = 1.0
