import gymnasium as gym
import numpy as np

from util.constants import Constants

class RacerEnv:
    metadata = {} 
    def __init__(self):
        # Values from https://github.com/Tinker-Twins/AutoDRIVE-F1TENTH-ARMLab/blob/main/Car-Parameters.md or https://github.com/CPS-TUWien/racing_dreamer/blob/398970bf2b4bf167cf53c0d0b0128a1b63d3377d/ros_agent/agents/follow_the_gap/src/agent.py#L72
        
        constants = Constants()
        self._min_steering = constants.min_steering
        self._max_steering = constants.max_steering
        self._min_speed = constants.min_speed
        self._max_speed = constants.max_speed
    
    @property
    def observation_space(self):
        pass

    def denormalize(self, normalized_value: float, min_physical_value: float, max_physical_value: float) -> float:
        """
        Denormalize a value from the range [-1, 1] to the range [min_physical_value, max_physical_value]
        
        Args:
            normalized_value: The normalized value to denormalize
            min_physical_value: The minimum value of the physical range
            max_physical_value: The maximum value of the physical range
        
        Returns:
            The denormalized value
        """
        mean_physical_value = (min_physical_value + max_physical_value) / 2
        physical_amplitude = (max_physical_value - min_physical_value) / 2
        return mean_physical_value + normalized_value * physical_amplitude

    @property
    def action_space(self):
        return gym.spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32),
                              high=np.array([1.0, 1.0], dtype=np.float32),
                              shape=(2,),
                              dtype=np.float32)


    def step(self, action):
        pass
      # Denormalize action to physical control values
        steering = self.denormalize(action[0], self._min_steering, self._max_steering)
        speed = self.denormalize(action[1], self._min_speed, self._max_speed)
      # Use steering and speed for simulation
        observation = self.get_observation()
        reward = self.calculate_reward()
        done = self.check_done()
        info = {}
        
        return observation, reward, done, info

    def reset(self):
        pass