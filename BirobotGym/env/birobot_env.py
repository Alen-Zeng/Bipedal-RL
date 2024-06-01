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
                # height_reward_weight:float = 0.2,
                # forward_reward_weight: float = 2,
                # ctrl_cost_weight: float = 0.5,
                # contact_cost_weight: float = 5e-6,
                # angular_reward_weight:float = 2e-2,
                # contact_cost_range: Tuple[float, float] = (-np.inf, 5.0),
                # healthy_reward: float = 1.0,
                # terminate_when_unhealthy: bool = True,
                # healthy_z_range: Tuple[float, float] = (0.3, 0.55),
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
        self.healthy_z_range = (0.3, 0.55)
        self.height_reward_weight = 1.
        self.tracking_lin_vel_weight = 1.
        self.lin_vel_yz_weight = -1.
        self.angular_vel_reward_weight = -1.
        self.no_fly_weight = 1.
        # self.forward_reward_weight = 
        # self.ctrl_cost_weight = 
        # self.contact_cost_weight = 
        # self.contact_cost_range = 
        # self.terminate_when_unhealthy = 

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
        self.render_mode = "human"

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(291,), dtype=np.float64)

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

        # print("\033[2J")
        # print(
        #     "reward_survive",reward_info["reward_survive"],
        #     "\nreward_forward",reward_info["reward_forward"],
        #     "\nreward_ctrl",reward_info["reward_ctrl"],
        #     "\nreward_contact",reward_info["reward_contact"],
        #     "\nangular_reward",reward_info["angular_reward"],
        #     "\naction",action
        #     )
        # print(":::",self.data.xpos[1])
        # print("geom1",self.data.contact.geom1)
        # print("geom2",self.data.contact.geom2)
        # print("cfrc_ext",self.data.cfrc_ext[6])
        # print("cfrc_ext",self.data.cfrc_ext[11])
        # print("no fly:",self.get_if_no_fly)
        # print("=====================")

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
        reward_tracking_ang_vel = 1.
        reward_torques = -5.e-6
        reward_dof_vel = -0.
        reward_dof_acc = -2.e-7
        reward_feet_air_time =  5.
        reward_collision = -1.
        reward_action_rate = 0.01
        reward_dof_pos_limits = -1.
        reward_feet_contact_forces = -0.
######*******

        reward_healthy = self.get_healthy_reward  # 保持不瘫倒的奖励
        reward_termination = 0. #终端被重置的惩罚
        if not self.not_healthy_terminated:
            reward_termination = -200.
        reward_height = self.height_reward_weight * (-2*pow((self.data.xpos[1][2] - 4.1),2) + 1) # 保持一定高度奖励
        reward_xvel = self.tracking_lin_vel_weight*(-pow((x_velocity - 1.5),2) + 1) # X方向速度奖励
        reward_yzvel = self.lin_vel_yz_weight*(y_velocity+z_velocity) # 惩罚Y方向和Z方向的速度
        reward_angular_vel = self.angular_vel_reward_weight *np.sum(np.square(angular_vel)) # 惩罚躯干的角速度
        reward_no_fly = self.get_if_no_fly*self.no_fly_weight  #没有腾空的奖励

        


######






        
        # forward_reward = -self._forward_reward_weight * np.square(self.data.xpos[1][1])

        # healthy_reward = self.healthy_reward
        # print("angular_reward",angular_reward)
        # rewards = xvel_reward + yzvel_reward + healthy_reward+angular_reward+forward_reward+height_reward
        reward = 0

        # ctrl_cost = self.control_cost(action)
        # contact_cost = self.contact_cost
        # costs = ctrl_cost + contact_cost
        # print("costs",costs)

        # reward = rewards - costs

        reward_info = {
            # "height_reward": height_reward,
            # "forward_reward": forward_reward,
            # "xvel_reward": xvel_reward,
            # "yzvel_reward": yzvel_reward,
            # "healthy_reward": healthy_reward,
            # "angular_reward": angular_reward,
            # "reward_ctrl": -ctrl_cost,
            # "reward_contact": -contact_cost,
            # "reward":reward,
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


    # def control_cost(self, action):
    #     control_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
    #     return control_cost
    
    # @property
    # def contact_cost(self):
    #     contact_forces = self.data.cfrc_ext
    #     contact_cost = self._contact_cost_weight * np.sum(np.square(contact_forces))
    #     min_cost, max_cost = self._contact_cost_range
    #     contact_cost = np.clip(contact_cost, min_cost, max_cost)
    #     return contact_cost
    
