import gymnasium as gym
import numpy as np

from util.constants import Constants


class RacerEnv:
    metadata = {}

    def __init__(self):
        # Values from https://github.com/Tinker-Twins/AutoDRIVE-F1TENTH-ARMLab/blob/main/Car-Parameters.md or https://github.com/CPS-TUWien/racing_dreamer/blob/398970bf2b4bf167cf53c0d0b0128a1b63d3377d/ros_agent/agents/follow_the_gap/src/agent.py#L72

        constants = Constants()

        self.min_lidar = constants.min_lidar  # 0.021
        self.max_lidar = constants.max_lidar  # 30.0
        self.num_lidar = constants.num_lidar  # 1080

        self._min_steering = constants.min_steering
        self._max_steering = constants.max_steering
        self._min_speed = constants.min_speed
        self._max_speed = constants.max_speed

    @property
    def observation_space(self, include_odom=False):
        """
        returns a dictionary containing LiDAR data & odomoetry data (optional)
        """
        self.include_odom = include_odom
        self.lidar_space = gym.spaces.Box(
            low=np.array([self.min_lidar] * self.num_lidar, dtype=np.float32),
            high=np.array([self.max_lidar] * self.num_lidar, dtype=np.float32),
            shape=(self.num_lidar,),
            dtype=np.float32,
        )

        # odom data: [x, y, z, linear_velocity, angular_velocity]
        if self.include_odom:
            self.odom_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
            )
        else:
            self.odom_space = None

        spaces = {"lidar": self.lidar_space}
        if self.include_odom:
            spaces["odom"] = self.odom_space

        return gym.spaces.Dict(spaces)

    def denormalize(
        self,
        normalized_value: float,
        min_physical_value: float,
        max_physical_value: float,
    ) -> float:
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
        return gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

    def step(self, action, lidar_data, odom_data=None):
        # Denormalize action to physical control values
        steering = self.denormalize(action[0], self._min_steering, self._max_steering)
        speed = self.denormalize(action[1], self._min_speed, self._max_speed)
        # Use steering and speed for simulation
        lidar_data = np.clip(lidar_data, self.lidar_space.low, self.lidar_space.high)
        observation = {"lidar": lidar_data}
        if self.include_odom and odom_data != None:
            observation["odom"] = np.array(odom_data, dtype=np.float32)

        # TODO: update reward function
        reward = None
        done = False
        info = {}

        return observation, reward, done, info

    def reset(self, seed=None, options=None, lidar_data=None, odom_data=None):
        super().reset(seed=seed)
        if lidar_data is None:
            raise ValueError("Reset: LiDAR data does not exist")

        lidar_data = np.clip(lidar_data, self.lidar_space.low, self.lidar_space.high)

        observation = {"lidar": lidar_data}
        if self.include_odom and odom_data != None:
            observation["odom"] = np.array(odom_data, dtype=np.float32)

        info = {}
        return observation, info
