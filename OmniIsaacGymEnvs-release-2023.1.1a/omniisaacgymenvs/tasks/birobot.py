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
        pass# TODO

    
    def get_anymal(self):
        birobot = Birobot(
            prim_path=self.default_zero_env_path + "/Birobotisaac",name="Birobot",translation=self._birobot_translation
            )
        self._sim_config.apply_articulation_settings(
            "Birobot", get_prim_at_path(birobot.prim_path), self._sim_config.parse_actor_config("Birobot")
        )

        # TODO 在这里使用set_drive 配置机器人关节驱动的参数