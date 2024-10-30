import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class SimpleNode(Node):
    def __init__(self):
        super().__init__('simple_node')
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

    def lidar_callback(self, msg):
        self.get_logger().info(f"Received LaserScan data: {msg.ranges[0]}")

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.steering_angle = 0.0
        drive_msg.drive.speed = 1.0
        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init()
    node = SimpleNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

