from gymnasium.envs.registration import register

register(
    id="Birobot-v0",
    entry_point="BirobotGym.env:Birobot",
    max_episode_steps=150000,
)