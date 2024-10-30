#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

class SafetyNode(Node):

    def __init__(self):
        super().__init__('safety_node')
        self.get_logger().info('Safety node has started.')
        self.speed = 0.0
        self.publisher_ = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        
        self.subscription_odom = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.odom_callback,
            10)
        
        self.subscription_scan = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)

    def odom_callback(self, odom_msg):
        self.speed = odom_msg.twist.twist.linear.x

    def scan_callback(self, scan_msg):
        min_ttc = float('inf')
        for i, range in enumerate(scan_msg.ranges):
            if range > 0:
                angle = scan_msg.angle_min + i * scan_msg.angle_increment
                relative_speed = self.speed * np.cos(angle)
                if relative_speed > 0:
                    ttc = range / relative_speed
                    if ttc < min_ttc:
                        min_ttc = ttc

        if min_ttc < 1.0:
            brake_msg = AckermannDriveStamped()
            brake_msg.drive.speed = 0.0
            brake_msg.drive.acceleration = -5.0  
            self.publisher_.publish(brake_msg)
            self.get_logger().info('Brake msg is published: stopping the vehicle.')

def main(args=None):
    rclpy.init(args=args)
    safety_node = SafetyNode()
    rclpy.spin(safety_node)

    safety_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

