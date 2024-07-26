import math

import numpy as np
import torch
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.rotations import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.birobot import Birobot
from omni.isaac.core.articulations import ArticulationView
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive
from omni.isaac.core.prims import RigidPrimView

class BirobotTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self.update_config(sim_config)
        self._num_observations = 42 
        self._num_actions = 10   # def num_actions(self): return self._num_actions

        RLTask.__init__(self, name, env)
        return
    
    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # TODO 其他参数导入和配置

        # normalization
        self.lin_vel_scale = self._task_cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self._task_cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self._task_cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self._task_cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self._task_cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self._task_cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["ang_vel_z"] = self._task_cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["lin_vel_z"] = self._task_cfg["env"]["learn"]["linearVelocityZRewardScale"]
        self.rew_scales["joint_acc"] = self._task_cfg["env"]["learn"]["jointAccRewardScale"]
        self.rew_scales["action_rate"] = self._task_cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["cosmetic"] = self._task_cfg["env"]["learn"]["cosmeticRewardScale"]
        self.rew_scales["footairtime"] = self._task_cfg["env"]["learn"]["footairtimeRewardScale"]
        self.rew_scales["nofly"] = self._task_cfg["env"]["learn"]["noflyRewardScale"]
        self.rew_scales["base_parallel_ground"] = self._task_cfg["env"]["learn"]["base_parallel_ground"]

        # command ranges
        self.command_x_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # base init state
        pos = self._task_cfg["env"]["baseInitState"]["pos"]
        rot = self._task_cfg["env"]["baseInitState"]["rot"]
        v_lin = self._task_cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self._task_cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang
        self.base_init_state = state

        # default joint positions
        self.named_default_joint_angles = self._task_cfg["env"]["defaultJointAngles"]

        # joint driver
        self.Kp = self._task_cfg["env"]["control"]["stiffness"]
        self.Kd = self._task_cfg["env"]["control"]["damping"]

        # simulation / training
        self.dt = 1. / 60.
        self.max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._birobot_translation = torch.tensor([0.0 , 0.0 , 0.0])

        


    def set_up_scene(self, scene) -> None:
        self.get_birobot()
        super().set_up_scene(scene)
        self._birobot = ArticulationView(
            prim_paths_expr = "/World/envs/.*/Birobot3/Base_Link/Base_Link",name="birobotview", reset_xform_properties=False
        )# TODO 正则表达式是怎么确定的
        scene.add(self._birobot)

        # 添加脚部，追踪碰撞
        self.Lankle_prim = RigidPrimView(prim_paths_expr="/World/envs/.*/Birobot3/Base_Link/Lankle_Link0",name="Lankle_prim",track_contact_forces=True)
        self.Rankle_prim = RigidPrimView(prim_paths_expr="/World/envs/.*/Birobot3/Base_Link/Rankle_Link0",name="Rankle_prim",track_contact_forces=True)
        scene.add(self.Lankle_prim)
        scene.add(self.Rankle_prim)

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("birobotview"):
            scene.remove_object("birobotview", registry_only=True)
        self._birobot = ArticulationView(
            prim_paths_expr = "/World/envs/.*/Birobot3/Base_Link/Base_Link",name="birobotview", reset_xform_properties=False
        )
        scene.add(self._birobot)

    def get_birobot(self):
        birobot = Birobot(
            prim_path=self.default_zero_env_path + "/Birobot3",name="Birobot",translation=self._birobot_translation
            )
        self._sim_config.apply_articulation_settings(
            "Birobot", get_prim_at_path(birobot.prim_path), self._sim_config.parse_actor_config("Birobot")
        )

        #           prim_path,                                                  drive_type,  target_type, target_value, stiffness, damping, max_force
        set_drive("/World/envs/env_0/Birobot3/Base_Link/joints/Lhipyaw_Joint",  "angular",   "position",  0,            self.Kp,   self.Kd,      1600)
        set_drive("/World/envs/env_0/Birobot3/Base_Link/joints/Lhiproll_Joint", "angular",   "position",  0,            self.Kp,   self.Kd,      1600)
        set_drive("/World/envs/env_0/Birobot3/Base_Link/joints/Lthigh_Joint",   "angular",   "position",  0,            self.Kp,   self.Kd,      1600)
        set_drive("/World/envs/env_0/Birobot3/Base_Link/joints/Lknee_Joint0",   "angular",   "position",  0,            self.Kp,   self.Kd,      1600)
        set_drive("/World/envs/env_0/Birobot3/Base_Link/joints/Lankle_Joint0",  "angular",   "position",  0,            self.Kp,   self.Kd,      1600)

        set_drive("/World/envs/env_0/Birobot3/Base_Link/joints/Rhipyaw_Joint",  "angular",   "position",  0,            self.Kp,   self.Kd,      1600)
        set_drive("/World/envs/env_0/Birobot3/Base_Link/joints/Rhiproll_Joint", "angular",   "position",  0,            self.Kp,   self.Kd,      1600)
        set_drive("/World/envs/env_0/Birobot3/Base_Link/joints/Rthigh_Joint",   "angular",   "position",  0,            self.Kp,   self.Kd,      1600)
        set_drive("/World/envs/env_0/Birobot3/Base_Link/joints/Rknee_Joint0",   "angular",   "position",  0,            self.Kp,   self.Kd,      1600)
        set_drive("/World/envs/env_0/Birobot3/Base_Link/joints/Rankle_Joint0",  "angular",   "position",  0,            self.Kp,   self.Kd,      1600)


    def get_observations(self) -> dict:
        base_position, base_rotation = self._birobot.get_world_poses(clone=False)
        root_velocities = self._birobot.get_velocities(clone=False)
        dof_pos = self._birobot.get_joint_positions(clone=False)
        dof_vel = self._birobot.get_joint_velocities(clone=False)

        velocity = root_velocities[:, 0:3]
        ang_velocity = root_velocities[:, 3:6]

        base_lin_vel = quat_rotate_inverse(base_rotation, velocity) * self.lin_vel_scale
        base_ang_vel = quat_rotate_inverse(base_rotation, ang_velocity) * self.ang_vel_scale
        projected_gravity = quat_rotate(base_rotation, self.gravity_vec)  # 将重力向量self.gravity_vec从世界坐标系转换到躯干坐标系
        dof_pos_scaled = (dof_pos - self.default_dof_pos) * self.dof_pos_scale
        
        commands_scaled = self.commands * torch.tensor(
            [self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
            requires_grad=False,
            device=self.commands.device,
        )

        # TODO DEBUG
        # print("Lankle_prim force: ",self.Lankle_prim.get_net_contact_forces())
        # print("Rankle_prim force: ",self.Rankle_prim.get_net_contact_forces())


        self.obs_buf = torch.cat(
            (
                base_lin_vel,
                base_ang_vel,
                projected_gravity,
                commands_scaled,
                dof_pos_scaled,
                dof_vel * self.dof_vel_scale,
                self.actions,
            ),
            dim=-1,
        )

        observations = {self._birobot.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self.world.is_playing():
            return
        
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        indices = torch.arange(self._birobot.count, dtype=torch.int32, device=self._device)
        self.actions[:] = actions.clone().to(self._device)
        current_targets = self.current_targets + self.action_scale * self.actions * self.dt
        self.current_targets[:] = tensor_clamp(
            current_targets, self.birobot_dof_lower_limits, self.birobot_dof_upper_limits
        )
        self._birobot.set_joint_position_targets(self.current_targets, indices)


    def reset_idx(self, env_ids):
        '''每次仿真的复位'''
        num_resets = len(env_ids)
        dof_pos = self.default_dof_pos[env_ids]
        dof_vel = torch_rand_float(-0.1, 0.1, (num_resets, self._birobot.num_dof), device=self._device)

        self.current_targets[env_ids] = dof_pos[:]

        # root_vel = torch.zeros((num_resets, 6), device=self._device)
        root_vel = torch_rand_float(-0.01, 0.01, (num_resets, 6), device=self._device)

        
        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._birobot.set_joint_positions(dof_pos, indices)
        self._birobot.set_joint_velocities(dof_vel, indices)

        self._birobot.set_world_poses(
            self.initial_root_pos[env_ids].clone(), self.initial_root_rot[env_ids].clone(), indices
        )
        self._birobot.set_velocities(root_vel, indices)

        self.commands_x[env_ids] = torch_rand_float(
            self.command_x_range[0], self.command_x_range[1], (num_resets, 1), device=self._device
        ).squeeze()
        self.commands_y[env_ids] = torch_rand_float(
            self.command_y_range[0], self.command_y_range[1], (num_resets, 1), device=self._device
        ).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(
            self.command_yaw_range[0], self.command_yaw_range[1], (num_resets, 1), device=self._device
        ).squeeze()

        # TODO DEBUG
        # print("commands_x",self.commands_x)

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0

    def post_reset(self):
        '''初始环境复位，环境启动后配置部分参数，函数只会在仿真前被调用一次'''

        self.default_dof_pos = torch.zeros(
            (self.num_envs, self.num_actions), dtype=torch.float, device=self.device, requires_grad=False
        )

        # TODO DEBUG
        print("Base_Link index:",self._birobot.get_body_index("Base_Link"))
        print("Lankle_Link0 index:",self._birobot.get_body_index("Lankle_Link0"))
        print("Rankle_Link0 index:",self._birobot.get_body_index("Rankle_Link0"))

        dof_names = self._birobot.dof_names
        for i in range(self.num_actions):
            name = dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        self.initial_root_pos, self.initial_root_rot = self._birobot.get_world_poses()
        self.current_targets = self.default_dof_pos.clone()

        dof_limits = self._birobot.get_dof_limits()  # unit: degrees 
        self.birobot_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.birobot_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)

        # init command
        self.commands = torch.zeros(self._num_envs, 3, dtype=torch.float, device=self._device, requires_grad=False)
        self.commands[:, 0] = 1
        self.commands_x = self.commands.view(self._num_envs, 3)[..., 0]
        self.commands_y = self.commands.view(self._num_envs, 3)[..., 1]
        self.commands_yaw = self.commands.view(self._num_envs, 3)[..., 2]


        self.extras = {} # no extra data

        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self._device).repeat((self._num_envs, 1))
        self.actions = torch.zeros(
            self._num_envs, self.num_actions, dtype=torch.float, device=self._device, requires_grad=False
        )
        self.last_dof_vel = torch.zeros(
            (self._num_envs, self.num_actions), dtype=torch.float, device=self._device, requires_grad=False
        )
        self.last_actions = torch.zeros(
            self._num_envs, self.num_actions, dtype=torch.float, device=self._device, requires_grad=False
        )

        # other params
        self.Lfeet_air_time = torch.zeros(self._num_envs,dtype=torch.float, device=self._device, requires_grad=False)
        self.Rfeet_air_time = torch.zeros(self._num_envs,dtype=torch.float, device=self._device, requires_grad=False)
        self.base_ori = torch.tensor([1.,0.,0.,0.], dtype=torch.float, device=self._device, requires_grad=False)

        # buffer to record timeout environments
        self.time_out_buf = torch.zeros_like(self.reset_buf)
        # randomize all envs
        indices = torch.arange(self._birobot.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        base_position, base_rotation = self._birobot.get_world_poses(clone=False)
        root_velocities = self._birobot.get_velocities(clone=False)
        dof_pos = self._birobot.get_joint_positions(clone=False)
        dof_vel = self._birobot.get_joint_velocities(clone=False)
        left_foot_contact_force = self.Lankle_prim.get_net_contact_forces(dt=self.dt)
        right_foot_contact_force = self.Rankle_prim.get_net_contact_forces(dt=self.dt)

        velocity = root_velocities[:, 0:3]
        ang_velocity = root_velocities[:, 3:6]

        base_lin_vel = quat_rotate_inverse(base_rotation, velocity)
        base_ang_vel = quat_rotate_inverse(base_rotation, ang_velocity)

        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.commands[:, 2] - base_ang_vel[:, 2])
        rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.rew_scales["lin_vel_xy"]
        rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.rew_scales["ang_vel_z"]

        rew_lin_vel_z = torch.square(base_lin_vel[:, 2]) * self.rew_scales["lin_vel_z"]
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - dof_vel), dim=1) * self.rew_scales["joint_acc"]
        rew_action_rate = (
            torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]
        )
        rew_feet_air_time = self.get_foot_air_time_reward(left_foot_contact_force,right_foot_contact_force) * self.rew_scales["footairtime"]
        rew_no_fly = torch.logical_or(left_foot_contact_force[:,2],right_foot_contact_force[:,2]) * self.rew_scales["nofly"]
        rew_base_parallel_ground = torch.exp(-(torch.sum(torch.square(base_rotation-self.base_ori),dim=1))) * self.rew_scales["base_parallel_ground"]

        total_reward = (rew_lin_vel_xy 
                       + rew_ang_vel_z 
                       + rew_joint_acc 
                       + rew_action_rate 
                       + rew_lin_vel_z 
                       + rew_feet_air_time 
                       + rew_no_fly 
                       + rew_base_parallel_ground)
        total_reward = torch.clip(total_reward, 0.0, None)

        self.fallen_over = self.is_base_below_threshold(threshold=0.38, ground_heights=0.0)
        # TODO DEBUG
        # print("fallen over",self.fallen_over)
        # print("base_rotation",base_rotation)
        # print("rew_base_parallel_ground",rew_base_parallel_ground)
        # print("rew_feet_air_time",rew_feet_air_time)

        total_reward[torch.nonzero(self.fallen_over)] = -1
        self.rew_buf[:] = total_reward.detach()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = dof_vel[:]

    def is_base_below_threshold(self, threshold, ground_heights):
        base_position, base_rotation = self._birobot.get_world_poses(clone=False)
        base_heights = base_position[:, 2]
        base_heights -= ground_heights
        
        # TODO DEBUG
        # print("base_heights",base_heights)

        return base_heights[:] < threshold

    def is_done(self) -> None:
        # reset agents
        time_out = self.progress_buf >= self.max_episode_length - 1

        self.reset_buf[:] = time_out | self.fallen_over  # TODO 配置其他可能得reset标志位

    def get_foot_air_time_reward(self,left_foot_contact_force,right_foot_contact_force):
        '''返回抬脚的奖励，不包含奖励系数'''
        left_if_contact = left_foot_contact_force[:,2]>0.1
        left_first_contact = (self.Lfeet_air_time > 0.) * left_if_contact
        self.Lfeet_air_time += self.dt
        # TODO DEBUG
        # print("Lfeet_air_time",self.Lfeet_air_time)
        left_rew_airTime = self.Lfeet_air_time * left_first_contact
        self.Lfeet_air_time *= ~left_if_contact
        
        right_if_contact = right_foot_contact_force[:,2]>0.1
        right_first_contact = (self.Rfeet_air_time > 0.) * right_if_contact
        self.Rfeet_air_time += self.dt
        # TODO DEBUG
        # print("Lfeet_air_time",self.Lfeet_air_time)
        right_rew_airTime = self.Rfeet_air_time * right_first_contact
        self.Rfeet_air_time *= ~right_if_contact

        return (left_rew_airTime + right_rew_airTime)