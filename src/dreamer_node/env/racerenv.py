import gymnasium as gym
import numpy as np


class RacerEnv(gym.Env):
    metadata = {}
    def __init__(self, min_lidar = 0.021, max_lidar = 30.0, num_lidar = 1080, inculde_odom = False):
        
        self.lidar_space = gym.spaces.Box(low = np.array([min_lidar] * num_lidar, dtype = np.float32),
                                          high = np.array([max_lidar] * num_lidar, dtype = np.float32),
                                          shape = (num_lidar,), dtype = np.float32)
        
        #odom data: [x, y, z, linear_velocity, angular_velocity]
        if self.include_odom:
            self.odom_space = gym.spaces.Box(low = -np.inf, high = np.inf, shape = (5,), dtype = np.float32)
        else:
            self.odom_space = None

    @property
    def observation_space(self):
        '''
        returns a dictionary containing LiDAR data & odomoetry data (optional)
        '''
        spaces = {"lidar": self.lidar_space}
        if self.include_odom:
            spaces["odom"] = self.odom_space
            
        return gym.spaces.Dict(spaces)      
        
    @property
    def action_space(self):
        pass
    
    def step(self, action, lidar_data, odom_data = None):
        
        steering = self.denormalize(action[0], self.min_sterring, self.max_steering)
        speed = self.denormalize(action[1], self.min_speed, self.max_speed)
        
        lidar_data = np.clip(lidar_data, self.lidar_space.low, self.lidar_space.high)
        
        observation = {"lidar": lidar_data}
        if self.include_odom and odom_data != None:
            observation["odom"] = np.array(odom_data, dtype = np.float32)

        reward = 0.0
        done = False
        info = {}
        
        return observation, reward, done, info
        

    def reset(self, seed = None, options = None, lidar_data = None, odom_data = None):
        super().reset(seed = seed)
        if lidar_data is None:
            raise ValueError("Reset: LiDAR data does not exist")
        
        lidar_data = np.clip(lidar_data, self.lidar_space.low, self.lidar_space.high)
        
        observation = {"lidar": lidar_data}
        if self.include_odom and odom_data != None:
            observation["odom"] = np.array(odom_data, dtype = np.float32)
            
        info = {}
        return observation, info

#whatsup
