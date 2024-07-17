from typing import Optional

import numpy as np
import torch
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import PhysxSchema

class Birobot(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "Birobot",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        
        self._usd_path = "/home/zxl/Documents/isaac/BipedalIsaac/Biroboturdf1.0/Birobot2/urdf/Birobot3/Birobot3.usd"
        self._name = name
        
        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )

        self._dof_names = [
            "Lhipyaw_Joint",
            "Lhiproll_Joint",
            "Lthigh_Joint",
            "Lknee_Joint0",
            "Lankle_Joint0",
            "Rhipyaw_Joint",
            "Rhiproll_Joint",
            "Rthigh_Joint",
            "Rknee_Joint0",
            "Rankle_Joint0"
        ]

    @property
    def dof_names(self):
        return self._dof_names
    

    