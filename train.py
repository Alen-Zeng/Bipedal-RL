import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import gymnasium as gym
import BirobotGym 
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from typing import Callable


def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init

def main():
    episode = 1000
    env_id = "Birobot-v0"
    num_cpu = 10  # Number of processes to use

    # Create the vectorized environment
    vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])


    # vec_env = make_vec_env("Birobot-v0",n_envs=5)

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

if __name__ == '__main__':
    main()