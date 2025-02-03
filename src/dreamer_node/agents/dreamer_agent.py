import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from typing import Tuple, List
import math
import os
import argparse
import ruamel.yaml as yaml
from ruamel.yaml import YAML
import pathlib
import gym
import torch
from util.constants import Constants
from util.constants import Config 
from dreamer.dream import Dreamer
import dreamer.tools as tools

class DreamerRacer(Node):
    """ 
    Implement Dreamer on the car
    """
    def __init__(self):
        super().__init__('dreamer_node')
        # constants
        self.const = Constants()
        self.config = Config() 

        # observations
        self.observations = dict()

        # pause on initial start
        self.speed = 0.0
        self.get_logger().info('Dreamer node has started')

        # variables
        self.time = self.get_clock().now()
        self.scan_num = 0

        # subscribers/publishers
        self.sub_scan = self.create_subscription(
            LaserScan,
            self.const.LIDAR_TOPIC,
            self.scan_callback, 
            10
        )
        
        self.pub_drive = self.create_publisher(
            AckermannDriveStamped, 
            self.const.DRIVE_TOPIC, 
            10
        )
        
        self.pub_scan = self.create_publisher(
            LaserScan,
            self.const.LIDAR_TOPIC,
            10
        )

        self.dreamer_init(self.config)


    def dreamer_init(self, config):
        tools.set_seed_everywhere(config.seed)
        if config.deterministic_run:
            tools.enable_deterministic_run()
        logdir = pathlib.Path(config.logdir).expanduser()
        config.traindir = config.traindir or logdir / "train_eps"
        config.evaldir = config.evaldir or logdir / "eval_eps"
        config.steps //= config.action_repeat
        config.eval_every //= config.action_repeat
        config.log_every //= config.action_repeat
        config.time_limit //= config.action_repeat
    
    def scan_callback(self, scan_msg: LaserScan):
        """
        Processes incoming LaserScan messages from the LiDAR sensor, 
        calculates appropriate driving commands (speed and steering angle), 
        and publishes these commands to the F1Tenth car's control interface.

        This function is the primary handler for LiDAR data and implements the 
        core logic for the car's autonomous navigation. It analyzes the 
        scan data to perceive the environment, detects obstacles, and 
        determines the optimal path to follow.  The calculated speed and 
        steering commands are then published to control the car's movement.

        Args:
            scan_msg: A LaserScan message containing the distance measurements 
                      from the LiDAR sensor.

        Returns:
            None.  The function publishes control messages directly.
        """

        self.observations["lidar"] = self.lidar_postproccess(scan_msg)

        steering = 0.0
        speed = 1.0

        # TODO: Set steering and speed variables
        drive_msg = self._convert_action(steering, speed)
        self.pub_drive.publish(drive_msg)

    def lidar_postproccess(self, lidar_data: LaserScan):
        """
        Processes raw LaserScan data to extract usable information for navigation.

        This function filters, extracts features (e.g., obstacles, distances), 
        or transforms the raw LiDAR data into a format suitable for path planning 
        or control.

        Args:
            lidar_data: The raw LaserScan message.

        Returns:
            The processed LiDAR data (e.g., distances, obstacle locations).
        """
        # calculate scan time stamp
        current_stamp = Time.from_msg(lidar_data.header.stamp)
        last_scan_time = current_stamp.nanoseconds - self.time.nanoseconds
        duration = Duration(nanoseconds=last_scan_time)
        
        if duration.nanoseconds / 1e9 < (0.08 - 0.001): # limit to approx. 10Hz
            assert f'Error: last laser scan time is {last_scan_time}. Limit = 10Hz'
            return
        
        # receive LiDAR scan rte
        self.time = current_stamp

        filtered_data = self._filter_range(lidar_data, self.const.FORWARD_SCAN_ARC)
        filtered_data = [round(point_range, 4) for point_range in filtered_data]

        if not hasattr(self, "last_log_time"):
            self.last_log_time = self.get_clock().now()
        
        elapsed_time = (self.get_clock().now() - self.last_log_time).nanoseconds / 1e9
        
        if elapsed_time >= 3:
            self.last_log_time = self.get_clock().now()
            print('dreamer received LIDAR scan @ rate:', last_scan_time / 1e9) # TODO: debug for scan rate
            print("observation = ", filtered_data); # TODO: debug range
        
        obs_lidar = filtered_data
        return obs_lidar 

    def _convert_action(self, steering_angle, speed) -> AckermannDriveStamped:
        """
        Converts steering/speed to AckermannDriveStamped.

        Creates an AckermannDriveStamped message from steering angle (rad) and speed.
        Prints the published action for debugging.

        Args:
            steering_angle: Desired steering angle (rad). Positive: left, negative: right.
            speed: Desired speed (m/s). Positive: forward.

        Returns:
            AckermannDriveStamped message.
        """
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        print('dreamer published action: steering_angle = ', steering_angle, "; speed = ", speed)
        return drive_msg

    def recursive_update(self, base, update): 
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                self.recursive_update(base[key], value)
            else:
                base[key] = value

    def _filter_range(self, lidar_data: LaserScan, range: Tuple[float]) -> List[float]:
        """
        Filters raw scan for only the desired arc.

        Args:
            lidar_data: original scan message
            range: the arc to filter for (e.g. (-pi/2, +pi/2))
        
        Returns:
            List of filtered ranges
        """
        raw_scan = list(np.flip(np.array(lidar_data.ranges, dtype=float))) # include liDAR observations

        raw_min_angle = lidar_data.angle_min
        raw_max_angle = lidar_data.angle_max
        target_min_angle, target_max_angle = range
        
        raw_arc_length = raw_max_angle - raw_min_angle
        number_of_ranges = len(raw_scan)
        number_of_ranges_per_radian = number_of_ranges / raw_arc_length

        skipped_min_angle = target_min_angle - raw_min_angle
        skipped_max_angle = raw_max_angle - target_max_angle

        min_index = math.ceil(skipped_min_angle * number_of_ranges_per_radian)
        max_index = math.floor(number_of_ranges - skipped_max_angle * number_of_ranges_per_radian)

        return raw_scan[min_index : max_index]

def main(args=None):
    rclpy.init(args=args)
    print("Dreamer Initialized")
    dreamer_node = DreamerRacer()
    rclpy.spin(dreamer_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    dreamer_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()