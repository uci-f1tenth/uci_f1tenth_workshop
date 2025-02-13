import gymnasium
import racecar_gym.envs.gym_api

env = gymnasium.make(
    id='SingleAgentAustria-v0',
    render_mode='human'
)

done = False
reset_options = dict(mode='grid')
obs, info = env.reset(options=reset_options)

while not done:
    action = env.action_space.sample()
    obs, rewards, terminated, truncated, states = env.step(action)
    done = terminated or truncated

env.close()