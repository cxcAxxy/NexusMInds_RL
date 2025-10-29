import os
import numpy as np
from datetime import datetime
import sys
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.configs import rslCfgPPO,envCfg
from rsl_rl.utils import class_to_dict

def train():

    envCfg=class_to_dict(envCfg)
    train_cfg = class_to_dict(rslCfgPPO())
    env = RealmanGraspEnv(envCfg)
    ppo_runner=OnPolicyRunner(env=env,train_cfg=train_cfg,log_dir='rsl_rl/logs')
    ppo_runner.learn(num_learning_iterations=train_cfg["runner"]["max_iterations"],init_at_random_ep_len=True)

if __name__ == '__main__':
    train()
