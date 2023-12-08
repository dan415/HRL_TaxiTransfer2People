from gym.envs.registration import register

register(
    id='custom/Taxi-v1.7',
    entry_point='custom.taxi:Taxi2PEnv',
    max_episode_steps=1000,
    reward_threshold=8
)
