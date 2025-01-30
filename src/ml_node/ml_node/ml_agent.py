#!/usr/bin/env python3
import numpy as np
import os

from ml_node.ml import ML_mod
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from sensor_msgs.msg import LaserScan

import torch

class MLAgent(Node):
    def __init__(self):
        super().__init__('ml_agent_node')
        self.is_real = False

        # Topics & Subs, Pubs
        self.lidarscan_topic = '/scan'  # /scan or /fake_scan
        drive_topic = '/drive'

        # Subscribe to scan
        self.sub_scan = self.create_subscription(LaserScan, self.lidarscan_topic, self.scan_callback, 10)
        self.sub_scan  # prevent unused variable warning
        # Publish to drive
        self.pub_drive = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.drive_msg = AckermannDriveStamped()

        # Timer
        self.timer = self.create_timer(0.1, self.timer_callback)  # timer period = 0.01s

        # MEGA-DAgger config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = ML_mod(observ_dim=108, hidden_dim=256, action_dim=2, lr=0.001, device=device)
        # TODO: change path
        package_share_directory = os.path.join(os.path.dirname(__file__), '..', 'resource')
        model_path = os.path.join(package_share_directory, 'mega_dagger.pkl')
        self.agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    def scan_callback(self, scan_msg):
        scan = np.array(scan_msg.ranges[::10]).flatten()  # 108
        if self.is_real and self.lidarscan_topic == '/scan':
            scan = scan[1:]
        # print(scan.shape)

        # NN input scan, output steering & speed
        agent_action = self.agent.get_action(scan)
        steering = float(agent_action[0])
        speed = float(agent_action[1])
        speed = min(float(agent_action[1]) / 2, 1.5)
        # speed = min(float(agent_action[1] / 3.0), 1.0)  # tuning this!
        # print(speed)

        # publish drive message
        self.drive_msg.drive.steering_angle = steering
        # self.drive_msg.drive.speed = (-1.0 if self.is_real else 1.0) * speed
        self.drive_msg.drive.speed = speed
        # self.pub_drive.publish(self.drive_msg)
        # print("steering = {}, speed = {}".format(round(steering, 5), round(speed, 5)))

    def timer_callback(self):
        self.pub_drive.publish(self.drive_msg)
        print("steering = {}, speed = {}".format(round(self.drive_msg.drive.steering_angle, 5), round(self.drive_msg.drive.speed, 5)))

def main(args=None):
    rclpy.init(args=args)
    print("ML Agent Initialized")
    agent_node = MLAgent()
    rclpy.spin(agent_node)

    agent_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

