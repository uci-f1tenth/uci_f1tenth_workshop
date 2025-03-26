import os
import sys
import pathlib
import functools

import numpy as np
from env.racecar_env import Racecar  # <-- Updated import


import numpy as np
from racecar_env import Racecar
import racecar_gym.envs.gym_api  # noqa: F401

import torch
from torch import nn
from torch import distributions as torchd

import tools
import models
import exploration as expl
from parallel import Parallel, Damy

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from util.constants import Config

def check_space_bounds(space, name):
    print(f"== {name} Space ==")
    print(f"  Shape: {space.shape}")
    print(f"  Dtype: {space.dtype}")
    if hasattr(space, 'low') and hasattr(space, 'high'):
        print(f"  Low:  {space.low}")
        print(f"  High: {space.high}")
    else:
        print("  No 'low' and 'high' attributes found")

def check_value_within_bounds(value, space, name):
    assert space.contains(value), f"{name} value is out of bounds!"
    if hasattr(space, 'low') and hasattr(space, 'high'):
        assert np.all(value >= space.low), f"{name} below lower bound!"
        assert np.all(value <= space.high), f"{name} above upper bound!"

def main():
    env = Racecar(train=True)

    # Check bounds of observation and action spaces
    check_space_bounds(env.observation_space, "Observation")
    check_space_bounds(env.action_space, "Action")

    # Reset the environment and check the observation
    obs = env.reset()
    print("\n[✓] Checking initial observation...")
    check_value_within_bounds(obs, env.observation_space, "Observation")

    # Take several random actions and verify values stay within bounds
    print("[✓] Running a few steps and checking bounds...")
    for i in range(10):
        action = env.action_space.sample()
        check_value_within_bounds(action, env.action_space, f"Action (step {i})")
        obs, reward, done, info = env.step(action)
        check_value_within_bounds(obs, env.observation_space, f"Observation (step {i})")
        if done:
            env.reset()

    env.close()
    print("\n✅ All checks passed.")

if __name__ == "__main__":
    main()
