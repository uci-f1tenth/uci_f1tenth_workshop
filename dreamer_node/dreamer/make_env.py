#AI placeholder skeleton code for make env file

import argparse
import os
import sys
import pathlib
import numpy as np
import torch
import gym
from racecar_gym.envs import MultiAgentRaceEnv, ChangingTrackMultiAgentRaceEnv, MultiAgentScenario

# Ensure the script can find the correct modules
sys.path.append(str(pathlib.Path(__file__).parent))

# Wrappers (Make sure you have these available)
import dreamer.wrappers as wrappers


def make_env(track, multi_track=False, action_repeat=1, rendering=True):
    """
    Create a RaceCar Gym environment.

    Parameters:
        - track (str): The name of the track (must match a YAML file in `scenarios/eval/`).
        - multi_track (bool): Whether to use the `ChangingTrackMultiAgentRaceEnv` for multi-track racing.
        - action_repeat (int): Number of times an action is repeated.
        - rendering (bool): Whether to enable rendering.

    Returns:
        - env: A wrapped RaceCar Gym environment.
    """
    print(f"🏎️ Creating RaceCar Gym Environment: {track} (Multi-Track: {multi_track})")

    # Load track scenario from YAML file
    scenario = MultiAgentScenario.from_spec(f'scenarios/eval/{track}.yml', rendering=rendering)

    # Create either a multi-track or single-track environment
    if multi_track:
        env = ChangingTrackMultiAgentRaceEnv(scenarios=[scenario], order='manual')
    else:
        env = MultiAgentRaceEnv(scenario=scenario)

    # Apply necessary wrappers
    env = wrappers.RaceCarWrapper(env)
    env = wrappers.FixedResetMode(env, mode='grid')
    env = wrappers.ActionRepeat(env, action_repeat)
    env = wrappers.ReduceActionSpace(env, low=[0.005, -1.0], high=[1.0, 1.0])
    env = wrappers.OccupancyMapObs(env)

    print("✅ Environment successfully created!")
    print("🔹 Action Space:", env.action_space)
    print("🔹 Observation Space:", env.observation_space)

    return env


def run_random_agent(env, num_steps=100):
    """
    Run a simple random agent in the RaceCar Gym environment.

    Parameters:
        - env: The environment instance.
        - num_steps (int): Number of steps to run.
    """
    print("\n🏁 Running Random Agent for", num_steps, "steps...")
    obs = env.reset()
    total_reward = 0

    for step in range(num_steps):
        action = env.action_space.sample()  # Take random action
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            print(f"🔄 Resetting environment at step {step}")
            obs = env.reset()

    print(f"\n🏆 Total Reward: {total_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone RaceCar Gym Dreamer Script")
    parser.add_argument("--track", type=str, default="track_name", help="Track name (YAML file in scenarios/eval/)")
    parser.add_argument("--multi_track", action="store_true", help="Enable multi-track mode")
    parser.add_argument("--action_repeat", type=int, default=1, help="Action repeat value")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps for random agent")
    args = parser.parse_args()

    # Make sure racecar_gym is installed
    try:
        import racecar_gym
    except ImportError:
        print("❌ racecar_gym not found. Install it with:")
        print("   pip install racecar-gym")
        sys.exit(1)

    # Create the environment
    env = make_env(args.track, args.multi_track, args.action_repeat, args.render)

    # Run a random agent for testing
    run_random_agent(env, num_steps=args.steps)
