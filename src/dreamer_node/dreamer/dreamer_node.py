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

        # pause on initial start
        self.speed = 0.0
        self.get_logger().info('Dreamer node has started')

        # variables
        self.time = self.get_clock().now()

        # subscribers/publishers
        self.subscription_scan = self.create_subscription(
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
        current_stamp = Time.from_msg(scan_msg.header.stamp)
        last_scan_time = current_stamp.nanoseconds - self.time.nanoseconds
        duration = Duration(nanoseconds=last_scan_time)
        if duration.nanoseconds / 1e9 < (0.08 - 0.001): # limit to approx. 10Hz
            assert f'Error: last laser scan time is {last_scan_time}. Limit = 10Hz'
            return
        
        self.time = current_stamp
        print('dreamer received LIDAR scan @ rate:', last_scan_time / 1e9) # TODO: debug for scan rate
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