import gymnasium as gym
import numpy as np

MIN_LIDAR = 0.021
MAX_LIDAR = 30.0

class RacerEnv:
    metadata = {}
    def __init__(self):
        pass

    @property
    def observation_space(self):
        '''
        returns a dictionary containing LiDAR data & odomoetry data (optional)
        '''
        
        self.lidar_space = gym.spaces.Box(low = np.array([MIN_LIDAR] * 1081, dtype = np.float32),
                                          high = np.array([MAX_LIDAR] * 1081, dtype = np.float32),
                                          shape = (1081,), dtype = np.float32)
        spaces = {"lidar": self.lidar_space}
        
        if self.include_odom:
            self.odom_space = gym.spaces.Box(low = -np.inf, high = np.inf, shape = (1081,), dtype = np.float32)
            spaces["odom"] = self.odom_space
        return gym.spaces.Dict(spaces)
        

    @property
    def action_space(self):
        pass

    def step(self, action, lidar_data, odom_data = None):
        #normalized_lidar = normalize_lidar(lidar_data)
        steering = self.denormalize(action[0], self.min_sterring, self.max_steering)
        speed = self.denormalize(action[1], self.min_speed, self.max_speed)
        
        lidar_data = np.clip(lidar_data, self.lidar_space.low, self.lidar_space.high)
        
        observation = {"lidar": lidar_data}
        if self.include_odom and odom_data != None:
            observation["odom"] = np.array(odom_data, dtype = np.float32)

        info = {}
        return observation, info
        

    def reset(self, seed = None, options = None, lidar_data = None, odom_data = None):
        if lidar_data is None:
            raise ValueError("Reset: LiDAR data does not exist")
        
        lidar_data = np.clip(lidar_data, self.lidar_space.low, self.lidar_space.high)
        
        observation = {"lidar": lidar_data}
        if self.include_odom and odom_data != None:
            observation["odom"] = np.array(odom_data, dtype = np.float32)
            
        info = {}
        return observation, info

#whatsup
