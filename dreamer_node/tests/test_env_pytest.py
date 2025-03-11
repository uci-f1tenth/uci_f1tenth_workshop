import pytest
import torch
import numpy as np
import warnings
import pathlib
import sys
from unittest.mock import MagicMock
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))  # Adds `dreamer_node`
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent / "dreamer")) 

# Import necessary modules from the provided script
from dreamer.racecar_env import Racecar
import tools
import models
import exploration as expl
from parallel import Parallel, Damy
from util.constants import Config
from dreamer import Dreamer  # Assuming this file is named dreamer.py

sys.path.append(str(pathlib.Path(__file__).parent.parent))

@pytest.fixture
def config():
    """Fixture to provide a default configuration."""
    return Config()

@pytest.fixture
def env():
    """Fixture to create a test environment."""
    env = Racecar(train=True)  # Use a mock environment if needed
    yield env
    env.close()

@pytest.fixture
def agent(env, config):
    """Fixture to create an instance of Dreamer."""
    logger = MagicMock()
    dataset = MagicMock()
    agent = Dreamer(env.observation_space, env.action_space, config, logger, dataset)
    return agent

def test_observation_space(env):
    """Ensure observations are valid and within range."""
    obs = env.reset()
    assert env.observation_space.contains(obs), "Observation space output is out of range"
    assert isinstance(obs, np.ndarray), "Observation should be a NumPy array"
    assert obs.shape == env.observation_space.shape, "Observation shape mismatch"

def test_action_space(env):
    """Ensure actions sampled from the space are valid."""
    action = env.action_space.sample()
    assert env.action_space.contains(action), "Action space output is out of range"

def test_step_function(env):
    """Ensure step function returns valid outputs."""
    obs = env.reset()
    action = env.action_space.sample()
    new_obs, reward, done, info = env.step(action)

    assert env.observation_space.contains(new_obs), "New observation out of range"
    assert isinstance(reward, (int, float)), "Reward is not a valid number"
    assert isinstance(done, bool), "Done flag should be a boolean"
    assert isinstance(info, dict), "Info should be a dictionary"

def test_no_warnings():
    """Ensure no warnings appear when running the environment."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        env = Racecar(train=True)
        assert len(w) == 0, f"Warnings found: {[str(warn.message) for warn in w]}"
        env.close()

def test_policy_output(agent, env):
    """Ensure the policy outputs valid actions."""
    obs = env.reset()
    reset = np.zeros(len(obs), dtype=bool)
    policy_output, state = agent(obs, reset, training=True)

    assert "action" in policy_output, "Policy output missing 'action'"
    assert "logprob" in policy_output, "Policy output missing 'logprob'"
    assert env.action_space.contains(policy_output["action"]), "Action out of bounds"

def test_train_function(agent):
    """Ensure the training function updates the agent."""
    data = MagicMock()  # Mock training data
    initial_update_count = agent._update_count
    agent._train(data)
    assert agent._update_count > initial_update_count, "Train function did not update agent"

def test_simulate_function(env, config):
    """Ensure simulation runs without issues."""
    train_eps = tools.load_episodes(config.traindir, limit=config.dataset_size)
    logger = MagicMock()
    
    state = tools.simulate(
        lambda o, d, s: (
            {"motor": torch.tensor([0.0]), "steering": torch.tensor([0.0])},
            None,
        ),
        [env],
        train_eps,
        config.traindir,
        logger,
        limit=config.dataset_size,
        steps=100,  # Small test batch
    )
    assert state is not None, "Simulate function failed"

