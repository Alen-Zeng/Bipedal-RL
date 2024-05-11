import math
from typing import Dict, Tuple, Union
import numpy as np
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle
from gymnasium.envs.mujoco import MujocoEnv

TODO = None

FPS=50
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}

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
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        
        #action space

        #observation space
        self.render_mode = "human"
        obs_size = self.data.qpos.size + self.data.qvel.size
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-1, high=1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-1, high=1, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
        

    def step(self, action):
        self.do_simulation(action,self.frame_skip)

        if self.render_mode == "human":
            self.render()

        reward = 0
        state = self._get_obs()
        # TODO
        # return state, reward,terminated:bool,truncated:bool,otherinfo TODO
        return state, reward,False,False,{}

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        observation = np.concatenate((position,velocity)).ravel()
        return observation

    def _get_info(self):
        pass

    
