# Import the gym_api module to trigger registration of your custom environment
from racecar_gym.envs import gym_api

# Import NumPy for working with numerical data and types
import numpy as np

# Import the custom Racecar environment wrapper for Dreamer
from dreamer.racecar_env import Racecar

# running tests in terminal with PYTHONPATH=dreamer_node pytest dreamer_node/tests/test_racecar_env.py

def test_reset_keys_and_types():
    # Initialize the environment in evaluation mode (train=False disables randomization or logging)
    env = Racecar(train=False)

    # Reset the environment to get the initial observation and info dictionary
    obs, info = env.reset()

    # Check that the observation is a dictionary
    assert isinstance(obs, dict)

    # Confirm that the observation contains an image key
    assert "image" in obs

    # The image should be of type uint8 (standard for RGB images)
    assert obs["image"].dtype == np.uint8

    # Check that the "is_first" key exists and is of type float32 and equals True on reset
    assert obs["is_first"].dtype == np.float32 and obs["is_first"][0] == True

    # Check that the "is_last" key exists, is float32, and is False on reset
    assert obs["is_last"].dtype == np.float32 and obs["is_last"][0] == False

    # Confirm that "is_terminal" exists and is float32 (used for learning termination signals)
    assert obs["is_terminal"].dtype == np.float32


def test_step_output_shapes():
    # Initialize the environment
    env = Racecar(train=False)

    # Reset the environment to get the initial observation
    obs, _ = env.reset()

    # Sample a valid random action from the environment's action space
    action = env.action_space.sample()

    # Some environments use Dict action spaces, so we support both ndarray and dict input formats
    output = env.step({"action": action}) if isinstance(action, np.ndarray) else env.step(action)

    # Unpack the output of a step
    obs, reward, done, truncated, info = output

    # Assertions to confirm the structure and types of the returned values
    assert isinstance(obs, dict)                 # Observation should be a dictionary
    assert isinstance(reward, np.float32)        # Reward should be a float32 scalar
    assert isinstance(done, bool)                # Done should be a boolean flag
    assert isinstance(truncated, bool)           # Truncated should be a boolean flag (e.g., time limit reached)
    assert isinstance(info, dict)                # Info should be a dictionary with debug metadata


def test_observation_space_structure():
    # Initialize the environment
    env = Racecar(train=False)

    # Get the observation space (a gymnasium.spaces.Dict)
    obs_space = env.observation_space

    # Check that all expected observation keys are in the Dict space
    assert "image" in obs_space.spaces
    assert "is_first" in obs_space.spaces
    assert "is_last" in obs_space.spaces
    assert "is_terminal" in obs_space.spaces

    # Verify the data types for some key observation fields
    assert obs_space.spaces["image"].dtype == np.uint8
    assert obs_space.spaces["is_first"].dtype == np.float32


def test_action_space_sampling():
    # Initialize the environment
    env = Racecar(train=False)

    # Sample a random action
    action = env.action_space.sample()

    # Check that the sampled action is valid within the action space
    assert env.action_space.contains(action)

