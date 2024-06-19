import math
from typing import Dict, Tuple, Union
import numpy as np
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle
from gymnasium.envs.mujoco import MujocoEnv
from stable_baselines3 import A2C
from Algorithm.PID.PID import DualLoopPID
from Algorithm.INV.INV import Leg_INV
import plotly.graph_objects as go



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

        self.last_action = np.zeros(10)

        self.healthy_weight = 1.
        self.healthy_z_range = (0.35, 0.55)
        self.height_reward_weight = 1.5
        self.tracking_lin_vel_weight = 1.5
        self.lin_vel_yz_weight = -1.
        self.angular_vel_reward_weight = -2.
        self.no_fly_weight = 1.
        self.ctrl_cost_weight = -2e-1
        self.collision_weight = -5e-8
        self.feet_air_time_weight = 8.
        self.joint_acc_weight=-2e-7
        self.foot_parallel_ground_weight = 1.
        self.base_parallel_ground_weight = 1.
        self.foot_step_weight = 1.

        # T, h, delta_h  https://www.mdpi.com/2076-3417/14/5/1803
        self.footstep_T = 0.8
        self.footstep_h = 0.3
        self.footstep_delta_h = 0.16

        #调PID test
        # self.ankle = []
        self.timearr = []
        # 画出高度图和抬腿图
        self.LH = []
        self.RH = []
        self.Ltheta1 = []
        self.Ltheta2 = []
        self.Ltheta3 = []
        self.Rtheta1 = []
        self.Rtheta2 = []
        self.Rtheta3 = []


        self.Lfeet_air_time = 0.
        self.Rfeet_air_time = 0.
        self.reward_healthy = 0

        self.L_foot_parallel_mat = np.array([0.707,0.707,0,
                                             0,0,1,
                                             0.707,-0.707,0])
        self.R_foot_parallel_mat = np.array([0.707,-0.707,0,
                                             0,0,-1,
                                             0.707,0.707,0])
        self.base_parallel_ground_mat = np.array([1.,0.,0.,
                                                  0.,1.,0.,
                                                  0.,0.,1.])


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
        #imitation 逆解算
        leftfoot_ref_height = self.zt_ref_left(self.data.time,self.footstep_T,self.footstep_h,self.footstep_delta_h)
        rightfoot_ref_height = self.zt_ref_right(self.data.time,self.footstep_T,self.footstep_h,self.footstep_delta_h)
        Ltheta1,Ltheta2,Ltheta3,Rtheta1,Rtheta2,Rtheta3 = self.leg_inv(leftfoot_ref_height,rightfoot_ref_height)
        actionINV = np.zeros_like(action)
        actionINV[2],actionINV[3],actionINV[4],actionINV[7],actionINV[8],actionINV[9] = Ltheta1,Ltheta2,Ltheta3,Rtheta1,Rtheta2,Rtheta3
        #低通滤波
        action[:] = 0.9*action[:]+0.1*self.last_action[:]
        self.last_action = action
        # RL补充的位置范围
        actionRL = np.full(len(action),1.) * action
        action[:] = actionRL[:] + actionINV[:]
        print("len:",len(action))
        action = self.joint_control(action,self.get_joint_pos(),self.get_joint_vel())

        # 画出高度图和抬腿图
        self.timearr.append(self.data.time)
        self.LH.append(leftfoot_ref_height)
        self.RH.append(rightfoot_ref_height)
        self.Ltheta1.append(Ltheta1)
        self.Ltheta2.append(Ltheta2)
        self.Ltheta3.append(Ltheta3)
        self.Rtheta1.append(Rtheta1)
        self.Rtheta2.append(Rtheta2)
        self.Rtheta3.append(Rtheta3)
        if(20.01 > self.data.time > 20. ):
            fig = go.Figure()        
            # 添加数据到曲线图
            fig.add_trace(go.Scatter(x=self.timearr, y=self.ankle, mode='lines', name='timearr'))
            fig.add_trace(go.Scatter(x=self.LH, y=self.ankle, mode='lines', name='LH'))
            fig.add_trace(go.Scatter(x=self.RH, y=self.ankle, mode='lines', name='RH'))
            fig.add_trace(go.Scatter(x=self.Ltheta1, y=self.ankle, mode='lines', name='Ltheta1'))
            fig.add_trace(go.Scatter(x=self.Ltheta2, y=self.ankle, mode='lines', name='Ltheta2'))
            fig.add_trace(go.Scatter(x=self.Ltheta3, y=self.ankle, mode='lines', name='Ltheta3'))
            fig.add_trace(go.Scatter(x=self.Rtheta1, y=self.ankle, mode='lines', name='Rtheta1'))
            fig.add_trace(go.Scatter(x=self.Rtheta2, y=self.ankle, mode='lines', name='Rtheta2'))
            fig.add_trace(go.Scatter(x=self.Rtheta3, y=self.ankle, mode='lines', name='Rtheta3'))
            # 设置图表标题和标签
            fig.update_layout(
                title='joint_leg',
                xaxis_title='时间 (s)',
                yaxis_title='值'
            )
            fig.show()

        #调PID test
        # indexjoint = 5
        # action = np.zeros(10)*0.
        # if(self.data.time >= 1.5):
        #     action[indexjoint] = 4.
        # action = self.joint_control(action,self.get_joint_pos(),self.get_joint_vel())
        # self.ankle.append(self.get_joint_pos()[indexjoint])
        # self.timearr.append(self.data.time)
        # if(8.01 > self.data.time > 8. ):
        #     fig = go.Figure()
        #     # 添加数据到曲线图
        #     fig.add_trace(go.Scatter(x=self.timearr, y=self.ankle, mode='lines', name='Example Curve'))
        #     # 设置图表标题和标签
        #     fig.update_layout(
        #         title='PID',
        #         xaxis_title='时间 (s)',
        #         yaxis_title='值'
        #     )
        #     # 显示图表
        #     fig.show()


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
        reward,reward_info = self._get_rew(x_velocity=x_velocity,y_velocity=y_velocity,z_velocity=z_velocity,angular_vel=angular_vel,leftfoot_ref_height=leftfoot_ref_height,rightfoot_ref_height=rightfoot_ref_height,action=action)

        terminated = (not self.not_healthy_terminated)


        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, reward_info
        # return observation, reward, False, False, reward_info

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

    def _get_rew(self, x_velocity:float, y_velocity:float,z_velocity:float,angular_vel,leftfoot_ref_height:float,rightfoot_ref_height:float,action):

###### 一些没用到的reward
        # reward_tracking_ang_vel = 1.
        # reward_dof_vel = -0.
        # reward_dof_pos_limits = -1.
        # reward_feet_contact_forces = -0.
        # reward_action_rate = self.get_joint_vel()
######*******
        
        self.reward_healthy = self.get_healthy_reward  # 保持不瘫倒的奖励
        reward_termination = 0. #终端被重置的惩罚
        if not self.not_healthy_terminated:
            reward_termination = -100.
        reward_height = self.height_reward_weight * math.exp(-10*pow((self.data.xpos[1][2] - 0.39),2)) # 保持一定高度奖励
        reward_xvel = self.tracking_lin_vel_weight*math.exp(-10*pow((x_velocity - 0.7),2)) # X方向速度奖励
        reward_yzvel = -self.lin_vel_yz_weight*math.exp(-np.square(y_velocity+z_velocity)) # 惩罚Y方向和Z方向的速度
        reward_angular_vel = -self.angular_vel_reward_weight *math.exp(-np.sum(np.square(angular_vel))) # 惩罚躯干的角速度
        reward_no_fly = self.get_if_no_fly*self.no_fly_weight  #没有腾空的奖励
        reward_control = self.control_cost()    # 惩罚过度控制joint_acc_weight
        reward_collision = self.collision_cost  #惩罚过度碰撞
        reward_feet_air_time = self.feet_air_time_weight*(self.get_foot_air_time(id=0)+self.get_foot_air_time(id=1))    #奖励抬腿
        reward_joint_acc = self.joint_acc_weight*np.sum(np.square(self.get_joint_vel()/(self.dt)))  #惩罚关节加速度过大
        # 奖励足部和地面平行
        reward_foot_parallel_ground = self.foot_parallel_ground_weight*math.exp(-np.sum(np.square(np.append(np.array(self.data.xmat[6])-self.L_foot_parallel_mat,np.array(self.data.xmat[11]-self.R_foot_parallel_mat)))))
        # 奖励身体和地面平行
        reward_base_parallel_ground = self.base_parallel_ground_weight*math.exp(-np.sum(np.square(np.array(self.data.xmat[1])-self.base_parallel_ground_mat)))
        # 奖励脚步交替踏步动作
        reward_foot_step = self.foot_step_weight*math.exp(-50*(pow((self.data.xpos[6][2]-leftfoot_ref_height),2)+pow((self.data.xpos[11][2]-rightfoot_ref_height),2)))

######

        reward = self.reward_healthy+reward_termination+reward_height+reward_xvel+reward_yzvel+reward_angular_vel+reward_no_fly+reward_control+reward_collision+reward_feet_air_time+reward_joint_acc+reward_foot_parallel_ground+reward_base_parallel_ground+reward_foot_step

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
            "reward_foot_parallel_ground":reward_foot_parallel_ground,
            "reward_base_parallel_ground":reward_base_parallel_ground,
            "reward_foot_step":reward_foot_step,
            "time":self.data.time,
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
            return -2*self.healthy_weight
              
    def get_foot_contact_ground(self,id:int):
        '''id:0 left foot(6),1 right foot(11)'''
        if(id == 0):
            return (self.data.cfrc_ext[6][5]>5)   # 左脚的Z方向受力（世界坐标系）
        elif(id == 1):
            return (self.data.cfrc_ext[11][5]>5)    # 右脚的Z方向受力（世界坐标系）
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
    
    def zt_ref_left(self,t, T, h, delta_h):
        '''左脚高度reference  https://www.mdpi.com/2076-3417/14/5/1803'''
        return max(0, h * math.sin(2 * math.pi * (t / T)) - delta_h)

    def zt_ref_right(self,t, T, h, delta_h):
        '''右脚高度reference  https://www.mdpi.com/2076-3417/14/5/1803'''
        return max(0, h * math.sin(2 * math.pi * (t / T) + math.pi) - delta_h)

    def joint_control(self,position_setpoint,position_feedback,velocity_feedback):
        num_motors = 10
        kp_pos = np.array([15.,5.,48.,12.,22.,
                           15.,5.,48.,12.,22.])
        ki_pos = np.array([0.,0.,0.2,0.,0.,
                           0.,0.,0.2,0.,0.])
        kd_pos = np.array([0.08,0.2,0.05,0.05,0.,
                           0.08,0.2,0.05,0.05,0.])
        kp_vel = np.array([1.5,1.,3.,2.,1.,
                           1.5,1.,3.,2.,1.])
        ki_vel = np.ones(num_motors) * 0.
        kd_vel = np.ones(num_motors) * 0.
        max_output_pos = np.ones(num_motors) * 1000.
        max_output_vel = np.ones(num_motors) * 1000.
        integrator_threshold_pos = np.ones(num_motors) * 10.
        integrator_threshold_vel = np.ones(num_motors) * 1.
        pid_controller = DualLoopPID(kp_pos, ki_pos, kd_pos, kp_vel, ki_vel, kd_vel,
                             integrator_threshold_pos, integrator_threshold_vel,
                             max_output_pos, max_output_vel)
        torque_output = pid_controller.control(position_setpoint, position_feedback, velocity_feedback)
        # 输出归一化
        torque_output = torque_output / max_output_vel
        return torque_output
    
    def leg_inv(self,LH,RH):
        leg_inv = Leg_INV()
        theta1,theta2,theta3 = leg_inv.inv_leg(H=LH)
        Ltheta1,Ltheta2,Ltheta3 = theta1,-theta2,-theta3
        theta1,theta2,theta3 = leg_inv.inv_leg(H=RH)
        Rtheta1,Rtheta2,Rtheta3 = -theta1,theta2,theta3
        return Ltheta1,Ltheta2,Ltheta3,Rtheta1,Rtheta2,Rtheta3
