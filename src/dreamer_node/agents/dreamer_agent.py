import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

from tools.constants import Constants

class Dreamer(Node):
    """ 
    Implement Dreamer on the car
    """
    def __init__(self):
        super().__init__('dreamer_node')
        # constants
        self.const = Constants()

        # observations
        self.observations = dict()

        # pause on initial start
        self.speed = 0.0
        self.get_logger().info('Dreamer node has started')

        # variables
        self.time = self.get_clock().now()

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
        
        scan_noised = scan_msg
        scan_noised.ranges = np.flip(self.observations["lidar"])
        self.pub_scan.publish(scan_noised)

        # TODO: Port dreamer here:
        # agent_action = self.agent.get_action(scan)
        # steering = float(agent_action[0])
        # speed = float(agent_action[1])
        # speed = min(float(agent_action[1]) / 2, 1.5)

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
        # receive LiDAR scan rate
        self.time = current_stamp
        print('dreamer received LIDAR scan @ rate:', last_scan_time / 1e9) # TODO: debug for scan rate

        raw_data = list(np.flip(np.array(lidar_data.ranges, dtype=float)))
        assert len(raw_data['lidar']) > 0, "Missing observations. Are you sure everything is linked properly?"
        print("observation = ", raw_data); # TODO: debug range

        obs_lidar = raw_data[0:1080,]
        extra_noise_stddev = 0.3 # 0.3m
        extra_noise = np.random.normal(0, extra_noise_stddev, 1080)
        return obs_lidar + extra_noise # adding noise (remove extra_noise to get rid of noise)

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

def main(args=None):
    rclpy.init(args=args)
    print("Dreamer Initialized")
    dreamer_node = Dreamer()
    rclpy.spin(dreamer_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    dreamer_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()