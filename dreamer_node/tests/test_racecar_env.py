import gymnasium as gym
import warnings
import numpy as np
from dreamer.racecar_env import Racecar

# Suppress irrelevant third-party warnings for cleaner test output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*RemovedInMarshmallow4Warning.*")
warnings.filterwarnings(
    "ignore",
    message=".*Conversion of an array with ndim > 0 to a scalar is deprecated.*",
)


# using the following to test console: PYTHONPATH=dreamer_node pytest -s dreamer_node/tests/test_racecar_env.py

# Test 1: Reset Observation Structure


def test_reset_keys_and_types():
    # Ensure that the first observation after reset has correct types and structure
    env = Racecar(train=False)
    obs, info = env.reset()

    assert isinstance(obs, dict)
    assert "image" in obs
    assert obs["image"].dtype == np.uint8
    assert obs["is_first"].dtype == np.float32
    assert obs["is_first"][0]

    assert obs["is_last"].dtype == np.float32
    assert obs["is_last"][0]
    assert obs["is_terminal"].dtype == np.float32


# ========================
# Test 2: Step Output Structure
# ========================


def test_step_output_shapes():
    # Make sure env.step() returns all expected values and in correct types
    env = Racecar(train=False)
    obs, _ = env.reset()
    action = env.action_space.sample()

    # Support for both dict and ndarray action formats
    output = (
        env.step({"action": action})
        if isinstance(action, np.ndarray)
        else env.step(action)
    )

    obs, reward, done, truncated, info = output
    assert isinstance(obs, dict)
    assert isinstance(reward, np.float32)
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


# ========================
# Test 3: Observation Space Structure
# ========================


def test_observation_space_structure():
    # Check if all required keys are declared in the observation space
    env = Racecar(train=False)
    obs_space = env.observation_space

    assert "image" in obs_space.spaces
    assert "is_first" in obs_space.spaces
    assert "is_last" in obs_space.spaces
    assert "is_terminal" in obs_space.spaces

    # Check for expected datatypes
    assert obs_space.spaces["image"].dtype == np.uint8
    assert obs_space.spaces["is_first"].dtype == np.float32


# ========================
# Test 4: Observation Values Within Space
# ========================


def test_observation_values_within_space():
    # Verify that values returned by the environment fall within the observation space bounds
    env = Racecar(train=False)
    # 🔍 Print observation space details
    print("\n🟦 Observation Space:")
    for key, space in env.observation_space.spaces.items():
        print(f"  - {key}: {space}")

    # 🔍 Print action space details
    print("\n🟨 Action Space:")
    print(env.action_space)

    obs_space = env.observation_space

    # Reset observation check
    obs, _ = env.reset()
    for key, space in obs_space.spaces.items():
        if key not in obs:
            warnings.warn(f"Missing optional key '{key}' in observation (reset)")
            continue
        value = obs[key]

        # If expected shape is (1,) but value is scalar, patch it
        if (
            isinstance(space, gym.spaces.Box)
            and space.shape == (1,)
            and np.shape(value) == ()
        ):
            value = np.array([value], dtype=space.dtype)

        assert np.all(np.isfinite(value)), (
            f"Non-finite value in '{key}' (reset)"
        )  # No NaNs or infs
        assert space.contains(value), (
            f"Value for '{key}' is out of bounds (reset)"
        )  # Value is in the defined Box or Discrete space

    # Step observation check
    action = env.action_space.sample()
    obs, *_ = env.step(action)
    for key, space in obs_space.spaces.items():
        if key not in obs:
            warnings.warn(f"Missing optional key '{key}' in observation (step)")
            continue
        value = obs[key]
        assert np.all(np.isfinite(value)), f"Non-finite value in '{key}' (step)"
        assert space.contains(value), f"Value for '{key}' is out of bounds (step)"


# ========================
# Test 5: Action Space Sampling
# ========================


def test_action_space_sampling():
    # Check that the environment's action space produces valid actions
    env = Racecar(train=False)
    action = env.action_space.sample()
    assert env.action_space.contains(action)
