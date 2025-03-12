import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import torch
from dreamer.dream import Dreamer
from dreamer.config import InferenceConfig, Ros2Config
from dreamer.racecar_env import Racecar
import racecar_gym.envs.gym_api  # noqa: F401


class Ros2Agent(Node):
    def __init__(self):
        super().__init__("dreamer_controller")

        # Initialize
        self.config = InferenceConfig()
        self.ros2config = Ros2Config()
        self.env = Racecar(train=False)
        self.agent = self.load_dreamer()
        self.state = None
        self.latest_scan = None
        self.obs = None

        # ROS2 setup
        self.lidar_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )

        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = np.clip(
            np.array(msg.ranges),
            self.ros2config.lidar_min_range,
            self.ros2config.lidar_min_range + self.ros2config.lidar_range,
        )

        self.latest_scan = self.latest_scan[: self.config.lidar_rays]

        self.obs, _ = self.env.observation_space  # Get structure from environment

        # Populate with real sensor data
        self.obs["lidar"] = np.nan_to_num(self.latest_scan)
        self.publish_actions()

    def publish_actions(self):
        # Call Dreamer
        with torch.no_grad():
            policy_output, self.state = self.agent(
                obs=self.obs, reset=np.array([False]), state=self.state, training=False
            )

        # TODO find unit for these
        action = policy_output["action"][0]
        control_msg = AckermannDriveStamped()
        control_msg.drive.speed = action[0] * self.ros2config.speed_conversion
        control_msg.drive.steering_angle = action[1] * self.ros2config.steer_conversion

        self.drive_pub.publish(control_msg)

    def load_dreamer(self):
        agent = Dreamer(
            obs_space=self.env.observation_space,
            act_space=self.env.action_space,
            config=self.config,
            logger=None,
            dataset=None,
        ).eval()

        logdir = pathlib.Path(self.config.logdir).expanduser()

        # Load trained weights
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])

        return agent.to(self.config.device)


def main(args=None):
    rclpy.init(args=args)
    racer = Ros2Agent()
    rclpy.spin(racer)

    racer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
