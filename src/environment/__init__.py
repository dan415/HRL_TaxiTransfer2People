
from gym.envs.registration import register

register(
     id='Taxi2p-v1',
     entry_point='src.environment:Taxi2PEnv',
     max_episode_steps=1000,
     reward_threshold=8

)