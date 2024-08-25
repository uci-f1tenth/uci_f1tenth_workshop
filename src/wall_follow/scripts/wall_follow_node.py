import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class WallFollow(Node):
    """ 
    Implement Wall Following on the car
    """
    def __init__(self):
        super().__init__('wall_follow_node')

        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # Subscribers and publishers
        self.scan_subscriber = self.create_subscription(LaserScan, lidarscan_topic, self.scan_callback, 10)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        # PID gains
        self.kp = 1.0
        self.kd = 0.1
        self.ki = 0.01

        # PID history
        self.integral = 0.0
        self.prev_error = 0.0

        # Desired distance from the wall
        self.desired_distance = 1.0

    def get_range(self, range_data, angle):
        """
        Simple helper to return the corresponding range measurement at a given angle. Make sure you take care of NaNs and infs.

        Args:
            range_data: single range array from the LiDAR
            angle: between angle_min and angle_max of the LiDAR

        Returns:
            range: range measurement in meters at the given angle
        """
        angle_index = int((angle - range_data.angle_min) / range_data.angle_increment)
        range_at_angle = range_data.ranges[angle_index]

        # Handle NaNs and infs
        if np.isinf(range_at_angle) or np.isnan(range_at_angle):
            range_at_angle = 10.0  # Arbitrary large distance if no obstacle detected

        return range_at_angle

    def get_error(self, range_data, dist):
        """
        Calculates the error to the wall. Follow the wall to the left (going counter clockwise in the Levine loop). You potentially will need to use get_range()

        Args:
            range_data: single range array from the LiDAR
            dist: desired distance to the wall

        Returns:
            error: calculated error
        """
        # Calculate the error using LIDAR data at specific angles
        theta = np.pi / 4  # 45 degrees

        range_90 = self.get_range(range_data, np.pi/2)  # Distance to the left wall
        range_theta = self.get_range(range_data, np.pi/2 + theta)  # Distance at 45 degrees to the front-left

        # Error based on the desired distance
        alpha = np.arctan((range_theta * np.cos(theta) - range_90) / (range_theta * np.sin(theta)))
        current_distance = range_90 * np.cos(alpha)

        error = dist - current_distance
        return error

    def pid_control(self, error, velocity):
        """
        Based on the calculated error, publish vehicle control

        Args:
            error: calculated error
            velocity: desired velocity

        Returns:
            None
        """
        # Calculate PID control
        self.integral += error
        derivative = error - self.prev_error

        control = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.prev_error = error

        # Create and publish drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = control
        drive_msg.drive.speed = velocity

        self.drive_publisher.publish(drive_msg)

    def scan_callback(self, msg):
        """
        Callback function for LaserScan messages. Calculate the error and publish the drive message in this function.

        Args:
            msg: Incoming LaserScan message

        Returns:
            None
        """
        error = self.get_error(msg, self.desired_distance)
        velocity = 1.0  # Fixed or dynamically adjusted speed

        self.pid_control(error, velocity)

def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    wall_follow_node = WallFollow()
    rclpy.spin(wall_follow_node)

    # Destroy the node explicitly
    wall_follow_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
