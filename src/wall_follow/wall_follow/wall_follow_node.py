import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

MIN_ANGLE = -2.3499999046325684
MAX_ANGLE = 2.3499999046325684
INCR_ANGLE = 0.004351851996034384

class WallFollow(Node):
    """ 
    Implement Wall Following on the car
    """
    def __init__(self):
        super().__init__('wall_follow_node')
        self.get_logger().info("Wall Following Node Started")

        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # TODO: create subscribers and publishers
        self.lidar_sub = self.create_subscription(LaserScan,
                                                  lidarscan_topic,
                                                  self.scan_callback,
                                                  10)
        
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               drive_topic,
                                               10)


        # TODO: set PID gains
        self.kp = 1.0
        self.kd = 0.1
        self.ki = 0.01

        # TODO: store history
        self.integral = 0.0
        self.prev_error = 0.0
        self.error = 0.0

        # TODO: store any necessary values you think you'll need

    def get_range(self, range_data, angle):
        """
        Simple helper to return the corresponding range measurement at a given angle. Make sure you take care of NaNs and infs.

        Args:
            range_data: single range array from the LiDAR
            angle: between angle_min and angle_max of the LiDAR

        Returns:
            range: range measurement in meters at the given angle

        """

        # angle_min & max = +- 135 degree
        # range = distance from obstacles (max_range = 30; the max dist from the obstacle is 30 meters)
        # Supppose the angle is given as rad
        range_index = int((angle - MIN_ANGLE)/INCR_ANGLE)
        range_at_angle = range_data[range_index]

        if np.isinf(range_at_angle) or np.isnan(range_at_angle):
            range_at_angle = 10.0  # Arbitrary large distance if no obstacle detected

        return range_at_angle
        # if 0 <= range_index < len(range_index):
        #     if not (np.isNaN(range_at_angle) or np.isinf(range_at_angle)):
        #         return range_at_angle
            
    def get_error(self, range_data, dist):
        """
        Calculates the error to the wall. Follow the wall to the left (going counter clockwise in the Levine loop). You potentially will need to use get_range()

        Args:
            range_data: single range array from the LiDAR
            dist: desired distance to the wall

        Returns:
            error: calculated error
        """

        theta = 50 * (np.pi/180) # theta set to 50 degrees


        #TODO:implement
        return 0.0

    def pid_control(self, error, velocity):
        """
        Based on the calculated error, publish vehicle control

        Args:
            error: calculated error
            velocity: desired velocity

        Returns:
            None
        """
        angle = 0.0
        # TODO: Use kp, ki & kd to implement a PID controller
        drive_msg = AckermannDriveStamped()
        # TODO: fill in drive message and publish

    def scan_callback(self, msg):
        """
        Callback function for LaserScan messages. Calculate the error and publish the drive message in this function.

        Args:
            msg: Incoming LaserScan message

        Returns:
            None
        """
        error = 0.0 # TODO: replace with error calculated by get_error()
        velocity = 0.0 # TODO: calculate desired car velocity based on error
        self.pid_control(error, velocity) # TODO: actuate the car with PID

        # Test
        test_range = self.get_range(msg, 0)
        self.get_logger().info(test_range)


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    wall_follow_node = WallFollow()
    rclpy.spin(wall_follow_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    wall_follow_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()