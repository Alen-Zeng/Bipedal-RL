import math
from typing import Dict, Tuple, Union
import numpy as np
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle
from gymnasium.envs.mujoco import MujocoEnv
from stable_baselines3 import A2C

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}

count = 0

def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()

class Birobot(MujocoEnv):
    metadata = {"render_modes":["human","rgb_array","depth_array",]}

    def __init__(self,
                xml_file:str="/home/zxl/Documents/Bipedal-RL/Biroboturdf1.0/Birobot2/urdf/Birobot3.xml",
                frame_skip:int=5,
                default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
                **kwargs,
                 ):

        EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            **kwargs
        )

        self.healthy_weight = 1.
        self.healthy_z_range = (0.35, 0.55)
        self.height_reward_weight = 1.
        self.tracking_lin_vel_weight = 2.
        self.lin_vel_yz_weight = -1.
        self.angular_vel_reward_weight = -3e-1
        self.no_fly_weight = 1.
        self.ctrl_cost_weight = -2e-1
        self.collision_weight = -5e-8
        self.feet_air_time_weight = 4.
        self.joint_acc_weight=-1e-7


        self.Lfeet_air_time = 0.
        self.Rfeet_air_time = 0.
        self.reward_healthy = 0


        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            # "render_fps": int(np.round(1.0 / self.dt)),
            "render_fps": int(26),
        }
        
        # 可视化
        self.render_mode = "rgb_array"
        # self.render_mode = "human"

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(311,), dtype=np.float64)

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.05, high=0.05, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0., high=0., size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
        

    def step(self, action):
        xyz_position_before = np.array(self.data.xpos[1])
        self.do_simulation(action,self.frame_skip)
        xyz_position_after = np.array(self.data.xpos[1])

        xyz_velocity = (xyz_position_after - xyz_position_before) / self.dt
        x_velocity = xyz_velocity[0]
        y_velocity = xyz_velocity[1]
        z_velocity = xyz_velocity[2]


        angular_vel = self.data.sensordata.flatten()
        angular_vel = np.array([angular_vel[0],angular_vel[1],angular_vel[2]])

        observation = self._get_obs()
        reward,reward_info = self._get_rew(x_velocity=x_velocity,y_velocity=y_velocity,z_velocity=z_velocity,angular_vel=angular_vel,action=action)

        terminated = (not self.not_healthy_terminated)


        if self.render_mode == "human":
            self.render()

        # print("reward info:",reward_info)

        return observation, reward, terminated, False, reward_info
        # return observation, reward, False, False, {}

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()
        angular_vel = self.data.sensordata.flatten()
        com_inertia = self.data.cinert[1:].flatten()
        com_velocity = self.data.cvel[1:].flatten()
        actuator_forces = self.data.qfrc_actuator[6:].flatten()
        external_contact_forces = self.data.cfrc_ext[1:].flatten()

        observation = np.concatenate((position,velocity,angular_vel,com_inertia,com_velocity,actuator_forces,external_contact_forces)).ravel()
        return observation

    def _get_rew(self, x_velocity:float, y_velocity:float,z_velocity:float,angular_vel,action):

######
        # reward_tracking_ang_vel = 1.
        # reward_dof_vel = -0.
        # reward_dof_pos_limits = -1.
        # reward_feet_contact_forces = -0.
        # reward_action_rate = self.get_joint_vel()
######*******
        
        self.reward_healthy = self.get_healthy_reward  # 保持不瘫倒的奖励
        reward_termination = 0. #终端被重置的惩罚
        if not self.not_healthy_terminated:
            reward_termination = -200.
        reward_height = self.height_reward_weight * (-pow((self.data.xpos[1][2] - 4.1),2) + 1) # 保持一定高度奖励
        reward_xvel = self.tracking_lin_vel_weight*(-pow((x_velocity - 1.5),2) + 1) # X方向速度奖励
        reward_yzvel = self.lin_vel_yz_weight*np.square(y_velocity+z_velocity) # 惩罚Y方向和Z方向的速度
        reward_angular_vel = self.angular_vel_reward_weight *np.sum(np.square(angular_vel)) # 惩罚躯干的角速度
        reward_no_fly = self.get_if_no_fly*self.no_fly_weight  #没有腾空的奖励
        reward_control = self.control_cost()    # 惩罚过度控制joint_acc_weight
        reward_collision = self.collision_cost  #惩罚过度碰撞
        reward_feet_air_time = self.feet_air_time_weight*(self.get_foot_air_time(id=0)+self.get_foot_air_time(id=1))    #奖励抬腿
        reward_joint_acc = self.joint_acc_weight*np.sum(np.square(self.get_joint_vel()/(self.dt)))  #惩罚关节加速度过大

######

        reward = self.reward_healthy+reward_termination+reward_height+reward_xvel+reward_yzvel+reward_angular_vel+reward_no_fly+reward_control+reward_collision+reward_feet_air_time+reward_joint_acc

        reward_info = {
            "reward_healthy":self.reward_healthy,
            "reward_termination":reward_termination,
            "reward_height":reward_height,
            "reward_xvel":reward_xvel,
            "reward_yzvel":reward_yzvel,
            "reward_angular_vel":reward_angular_vel,
            "reward_no_fly":reward_no_fly,
            "reward_control":reward_control,
            "reward_collision":reward_collision,
            "reward_feet_air_time":reward_feet_air_time,
            "reward_joint_acc":reward_joint_acc,
        }

        return reward, reward_info

    def _get_info(self):
        pass

    @property        
    def is_healthy(self):
        min_z, max_z = self.healthy_z_range
        is_healthy = min_z < self.data.xpos[1][2] < max_z

        return is_healthy
    
    @property        
    def not_healthy_terminated(self):
        min_z, max_z = self.healthy_z_range
        min_z -= 0.1
        max_z += 0.1
        not_healthy_terminated = min_z < self.data.xpos[1][2] < max_z

        return not_healthy_terminated
    
    @property
    def get_healthy_reward(self):
        if(self.is_healthy):
            return self.healthy_weight
        else:
            return -self.healthy_weight
              
    def get_foot_contact_ground(self,id:int):
        '''id:0 left foot(6),1 right foot(11)'''
        if(id == 0):
            return (self.data.cfrc_ext[6][5]>2)   # 左脚的Z方向受力（世界坐标系）
        elif(id == 1):
            return (self.data.cfrc_ext[11][5]>2)    # 右脚的Z方向受力（世界坐标系）
        else:
            return False
        
    @property
    def get_if_no_fly(self):
        if(self.get_foot_contact_ground(0) and (not self.get_foot_contact_ground(1))):
            return True
        elif(self.get_foot_contact_ground(1) and (not self.get_foot_contact_ground(0))):
            return True
        elif(self.get_foot_contact_ground(1) and  self.get_foot_contact_ground(0)):
            return True
        else:
            return False
        
    def get_foot_air_time(self,id:int):
        '''id:0 left foot,1 right foot,一个step要调用2次'''
        if(id == 0):
            if_contact = self.get_foot_contact_ground(id=id)
            first_contact = (self.Lfeet_air_time > 0.) * if_contact
            self.Lfeet_air_time += self.dt
            rew_airTime = np.sum((self.Lfeet_air_time) * first_contact) # reward only on first contact with the ground
            self.Lfeet_air_time *= ~if_contact
            return rew_airTime
        elif(id == 1):
            if_contact = self.get_foot_contact_ground(id=id)
            first_contact = (self.Rfeet_air_time > 0.) * if_contact
            self.Rfeet_air_time += self.dt
            rew_airTime = np.sum((self.Rfeet_air_time) * first_contact) # reward only on first contact with the ground
            self.Rfeet_air_time *= ~if_contact
            return rew_airTime
    

    def get_joint_pos(self):
        return self.data.sensordata[6:16]
    
    def get_joint_vel(self):
        return self.data.sensordata[16:]

    def control_cost(self):
        control_cost = self.ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
        return control_cost
    
    @property
    def collision_cost(self):
        collision_forces = self.data.cfrc_ext
        collision_cost = self.collision_weight * np.sum(np.square(collision_forces))
        min_cost, max_cost = (-np.inf, 10.0)
        collision_cost = np.clip(collision_cost, min_cost, max_cost)
        return collision_cost
    
