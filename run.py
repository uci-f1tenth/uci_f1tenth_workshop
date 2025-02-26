import gymnasium
import racecar_gym.envs.gym_api  # noqa: F401

env = gymnasium.make(
    id='SingleAgentRaceEnv-v0', 
    scenario='gyms/racecar_gym/scenarios/austria.yml',
    render_mode='human', # optional
    render_options=dict(width=320, height=240, agent='A') # optional
)

done = False
reset_options = dict(mode="grid")
obs, info = env.reset(options=reset_options)

print(env.action_space)

while not done:
    action = env.action_space.sample()
    # print("Actions")
    obs, rewards, terminated, truncated, states = env.step(action)
    # print("------")
    # print("Obs keys")
    # print(obs)
    # print("------")
    # print("State keys")
    # print(states.keys())
    done = terminated or truncated

env.close()
