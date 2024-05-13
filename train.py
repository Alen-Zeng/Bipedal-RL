import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import gymnasium as gym
import BirobotGym 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

episode = 1000
vec_env = make_vec_env("Birobot-v0",n_envs=1)

model = PPO("MlpPolicy",vec_env,verbose=1)
model.learn(total_timesteps=1)
print("Init model")
model.save("Birobottest1")
# del model
# model = PPO.load("Birobottest1")
print("start training")
for i in range(episode):
    model.learn(total_timesteps=10000)
    model.save("Birobottest1")
    print("episode NO.",i)

del model
model = PPO.load("Birobottest1")
print("finished training")
obs = vec_env.reset()
while True:
    action,_states = model.predict(obs)
    obs,rewards,dones,info = vec_env.step(action)