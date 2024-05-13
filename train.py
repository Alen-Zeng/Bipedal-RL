import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import gymnasium as gym
import BirobotGym 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

episode = 1000
vec_env = make_vec_env("Birobot-v0",n_envs=5)

model = PPO("MlpPolicy",vec_env,verbose=1)
model.learn(total_timesteps=1)
print("Init model")
model.save("modelPar/Birobot_epi0")
print("start training")
for i in range(episode):
    print("episode NO.",i)
    model.learn(total_timesteps=100000)
    if(i%10 == 0):
        modelname = "modelPar/Birobot_epi"+str(i)
        model.save(modelname)

del model
model = PPO.load("Birobottest1",vec_env)
print("finished training")
obs = vec_env.reset()
while True:
    action,_states = model.predict(obs)
    obs,rewards,dones,info = vec_env.step(action)