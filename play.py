import gymnasium as gym
import BirobotGym 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

vec_env = make_vec_env("Birobot-v0",n_envs=1)

model = PPO.load("modelPar/Birobot_epi440",vec_env)
print("start playing")
obs = vec_env.reset()
rew = 0
while True:
    action,_states = model.predict(obs)
    obs,rewards,dones,info = vec_env.step(action)
    
    if(dones):
        print("\033[2J")
        print("reward info:",
              "\nheight_reward:",info[0].get("height_reward"),
              "\nforward_reward:",info[0].get("forward_reward"),
              "\nxvel_reward:",info[0].get("xvel_reward"),
              "\nyzvel_reward:",info[0].get("yzvel_reward"),
              "\nhealthy_reward:",info[0].get("healthy_reward"),
              "\nangular_reward:",info[0].get("angular_reward"),
              "\nreward_ctrl:",info[0].get("reward_ctrl"),
              "\nreward_contact:",info[0].get("reward_contact"),
              "\nreward:",info[0].get("reward"),
              )
        rew = 0
    else:
        rew += rewards

    