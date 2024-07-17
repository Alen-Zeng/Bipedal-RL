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
        self._num_observations = 48 # TODO 具体怎么确定？
        self._num_actions = 10

        RLTask.__init__(self, name, env)
        return
    
    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # TODO 其他参数导入和配置
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
            prim_path=self.default_zero_env_path + "/Birobotisaac",name="Birobot",translation=self._birobot_translation
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