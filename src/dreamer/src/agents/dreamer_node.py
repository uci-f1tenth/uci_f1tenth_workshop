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

        # variables
        self.time = rclpy.Time(0)

        # subscribers/publishers
        self.subscription_scan = self.create_subsciprtion(
            LaserScan,
            self.const.LIDAR_TOPIC,
            self.scan_callback,  # run function
            10
        )
        self.pub_drive = self.create_publisher(
            AckermannDriveStamped, 
            self.const.DRIVE_TOPIC, 
            10
        )
    
    def scan_callback(self, scan_msg: LaserScan):
        last_scan_time = scan_msg.header.stamp - self.time
        if last_scan_time.to_sec() < (0.08 - 0.001): # limit to approx. 10Hz
            assert f'Error: last laser scan time is {last_scan_time}. Limit = 10Hz'
            return
        
        self.time = scan_msg.header.stamp
        print('dreamer received LIDAR scan @ rate:', last_scan_time.to_sec()) # TODO: debug for scan rate
        observations = dict()
        observations['lidar'] = np.flip(np.array(scan_msg.ranges)) # include liDAR observations
        print("observation = ", observations); # TODO: debug scan_msg.ranges

      
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