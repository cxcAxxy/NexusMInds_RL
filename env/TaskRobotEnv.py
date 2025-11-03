
from .core import RobotTaskEnv
from .Robot.gym_env.instance import *
from .Task import *

# 导入仿真的sim，封装好的Gym环境和MuJuCo 两个环境
from .Robot.gym_env.sim import Gym
#from Robot.MuJuCo_env.sim import MuJuCo

# 这个地方是根据我们提高的core.py提供的抽象类，然后根据Task定制的奖励函数设计，还有Robot提供的基于Isaac gym和MuJuCo等
# 仿真引擎等定制的机器人步进仿真平台
#这个地方的按照配置文件的方式进行处理

class FrankaReachGym(RobotTaskEnv):
    def __init__(self,cfg)-> None:

        #根据配置文件cfg，设置 sim的，robot，还有taskcfg。

        sim=Gym(cfg.gymcfg)
        robot=Franka(sim,cfg.robotcfg)
        task=Reach(sim,cfg.taskcfg)

        # init_buffer, 可能是需要存放一下其它的机器人信息，从仿真环境当中获取的。
        super().__init__(
            robot,
            task,
            cfg.all
        )

