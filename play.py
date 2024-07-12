import gymnasium as gym
import BirobotGym 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

vec_env = make_vec_env("Birobot-v0",n_envs=1)
modelnum = 3
modelname = "modelPar/Birobot_epi"+str(modelnum)
model = PPO.load(modelname,vec_env)
print("start playing")
obs = vec_env.reset()
rew = 0
while True:
    action,_states = model.predict(obs)
    obs,rewards,dones,info = vec_env.step(action)
    print("reward",rewards)
    print("info:",
            "\nreward_healthy",info[0].get("reward_healthy"),
            "\nreward_termination",info[0].get("reward_termination"),
            "\nreward_height",info[0].get("reward_height"),
            "\nreward_xvel",info[0].get("reward_xvel"),
            "\nreward_yzvel",info[0].get("reward_yzvel"),
            "\nreward_angular_vel",info[0].get("reward_angular_vel"),
            "\nreward_no_fly",info[0].get("reward_no_fly"),
            "\nreward_control",info[0].get("reward_control"),
            "\nreward_collision",info[0].get("reward_collision"),
            "\nreward_feet_air_time",info[0].get("reward_feet_air_time"),
            "\nreward_joint_acc",info[0].get("reward_joint_acc"),
            "\nreward_foot_parallel_ground",info[0].get("reward_foot_parallel_ground"),
            "\nreward_base_parallel_ground",info[0].get("reward_base_parallel_ground"),
            "\nreward_foot_step",info[0].get("reward_foot_step"),
            "\ntime",info[0].get("time"),
            )
    print("=====================================================")
    
    if(dones):
        # print("\033[2J")
        pass
    else:
        rew += rewards

    