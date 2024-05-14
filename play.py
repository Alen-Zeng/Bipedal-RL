import gymnasium as gym
import BirobotGym 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

vec_env = make_vec_env("Birobot-v0",n_envs=1)

model = PPO.load("modelPar/Birobot_epi50",vec_env)
print("start playing")
obs = vec_env.reset()
rew = 0
while True:
    action,_states = model.predict(obs)
    obs,rewards,dones,info = vec_env.step(action)
    
    if(dones):
        # print("\033[2J")
        print("reward:",rewards)
        rew = 0
    else:
        rew += rewards

    