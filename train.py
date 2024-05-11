from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

import gymnasium as gym
import BirobotGym 
# from BirobotGym.env.birobot_env import Birobot

envs = gym.vector.make("Birobot-v0",num_envs=2)
envs.reset()
action = np.array([(0.,0.,0.,1,0.,0.,0.,0.,0.,0.),(0.,0.,0.,1,0.,0.,0.,0.,0.,0.)])
while True:
    envs.step(actions=action)