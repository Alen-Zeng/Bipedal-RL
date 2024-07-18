import math

import numpy as np
import torch
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.rotations import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.birobot import Birobot
from omni.isaac.core.articulations import ArticulationView
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive

class BirobotTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self.update_config(sim_config)
        self._num_observations = 10 # TODO 配置真实的大小
        self._num_actions = 10

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
        self.dt = 1 / 60
        self.max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._birobot_translation = torch.tensor([0.0 , 0.0 , 0.4208])




    def set_up_scene(self, scene) -> None:
        self.get_birobot()
        super().set_up_scene(scene)
        self._birobot = ArticulationView(
            prim_paths_expr = "/World/envs/.*/Birobot3/Base_Link/Base_Link",name="birobotview", reset_xform_properties=False
        )# TODO 正则表达式是怎么确定的
        scene.add(self._birobot)

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

        # TODO 在这里使用set_drive 配置机器人关节驱动的参数

    def get_observations(self) -> dict:
        pass
        # TODO 配置获取相关的observation
        
        # self.obs_buf = 

    def pre_physics_step(self, actions) -> None:
        if not self.world.is_playing():
            return
        
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        
        #TODO 配置环境的复位

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        # TODO 还不知道这个函数有什么用
        pass

    def calculate_metrics(self) -> None:
        # TODO 各种参数的计算、REWARD的计算
        pass
        # self.rew_buf[:] = 

    def is_done(self) -> None:
        # reset agents
        time_out = self.progress_buf >= self.max_episode_length - 1
        self.reset_buf[:] = time_out  # TODO 配置其他可能得reset标志位