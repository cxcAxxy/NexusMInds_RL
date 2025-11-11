from .base_config import BaseConfig

class rslCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 1500  # number of policy updates

        # logging
        save_interval = 50  # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt

class LeggedRobotCfgDDPG(BaseConfig):
    seed = 1
    runner_class_name = 'OffPolicyRunner'

    class policy:
        init_noise_std = 0.1  # 减小初始噪声，提高稳定性
        actor_hidden_dims = [512, 256, 256 ,128 ]  # 使用标准DDPG网络结构
        critic_hidden_dims = [512, 256, 256, 128]  # 使用标准DDPG网络结构
        activation = 'relu'  # 使用ReLU激活函数
        n_critics = 1  # DDPG只需要一个Critic

    class algorithm:
        gamma = 0.99  # 提高折扣因子
        tau = 0.005  # 使用较小的软更新参数
        batch_size = 256  # 增加批量大小
        lr_actor = 1e-3  # 降低Actor学习率
        lr_critic = 1e-3  # Critic学习率可以稍高

    class runner:
        policy_class_name = 'DDPGActorCritic'
        algorithm_class_name = 'DDPG'
        num_steps_per_env = 50  # DDPG每步更新
        max_iterations = 2000  # 增加训练迭代数
        save_interval = 100
        experiment_name = 'DDPG_Panda'
        run_name = ''
        start_random_steps = 1000  # 增加随机探索步数
        max_size = int(1e6)  # 经验回放缓冲区大小

class envCfg(BaseConfig):
    pass