import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

from dreamer.src.constants import Constants

class Dreamer(Node):
    """ 
    Implement Dreamer on the car
    """
    def __init__(self):
        super().__init__('dreamer_node')
        # constants
        self.const = Constants()

        # pause on initial start
        self.speed = 0.0
        self.get_logger().info('Dreamer node has started')

        # publishers
        self.publisher_ = self.create_publisher(AckermannDriveStamped, self.const.DRIVE_TOPIC, 10)

        # subscribers
        self.subscription_odom = self.create_subscription(
            Odometry,
            self.const.ODOMETRY_TOPIC,
            self.odom_callback,
            10
        )

        self.subscription_scan = self.create_subsciprtion(
            LaserScan,
            self.const.LIDAR_TOPIC,
            self.scan_callback,
            10
        )

        # TODO: set PID gains
        # self.kp = 
        # self.kd = 
        # self.ki = 

        # TODO: store history
        # self.integral = 
        # self.prev_error = 
        # self.error = 

        # TODO: store any necessary values you think you'll need


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