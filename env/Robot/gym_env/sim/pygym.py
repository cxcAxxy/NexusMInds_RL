
import numpy as np
# gym应该要实现的接口

class Gym():
    def __init__(self):


    def pre_simulate(self,asset,robot_files,base_poses,base_ornes,num_envs):

        #这个地方设置neutral_joint_values，可以通过urdf或者是xml的关节角的限制确定，也可以通过外部的cfg设置得到
        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])

        # 设置摩擦等，基本的物理参数在这个地方，在导入actor的时候，要加入这个！

    def get_fingers_width(self):
        # 注意加入判断，是否有夹爪或者是灵巧手


    def control_joints(self,target_angles):
        # 给定目标关节角，使用pd去控制

    def get_ee_velocity(self):
        # 获得末端执行器的速度


    def get_ee_position(self):
        # 末端执行器

    def inverse_kinematics(link, position, orientation)
        # 逆运动学

    def get_joint_angle(self,joint_index):
        # 获得index的joint的数值

    def set_joint_neutral(self):
        # 设置初始的关节角的
        self.set_joint_angles(self.neutral_joint_values)

    def get_joint_angles(self):
        # 获得所有的关节角数值

    def set_joint_angles(self,target_joints):
        # 强制设置关节角的数值

    def set_base_pose(name, pos,orn):
        # 根据名字（这个很重要） 设置


    # 后面这些创建地板，还有其它的球体等，等参数根据mujuco或者gym的api自己增加，符合函数的要求即可
    def create_plane(self):

    def create_box(self):

    def create_sphere(self):

    def create_table(self):




    def step(self):