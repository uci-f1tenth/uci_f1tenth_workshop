import math
import pathlib
from typing import Tuple, List

import torch
from torch import distributions as torchd
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

import dreamer.tools as tools
from dreamer.config import Config
from dreamer.dream import Dreamer, Parallel, Damy


class Constants:
    """
    Class to hold constants for the car.
    """

    def __init__(self):
        self.FORWARD_SCAN_ARC = (np.deg2rad(-90.0), np.deg2rad(+90.0))
        # Topics
        self.LIDAR_TOPIC = "/scan"
        self.DRIVE_TOPIC = "/drive"
        self.ODOMETRY_TOPIC = "/ego_racecar/odom"

        # action space
        self.min_steering = -0.418
        self.max_steering = 0.418
        self.min_speed = 1.5
        self.max_speed = 19.67


class DreamerRacer(Node):
    """
    Implement Dreamer on the car
    """

    def __init__(self):
        super().__init__("dreamer_node")
        # constants
        self.const = Constants()
        self.config = Config()

        # observations
        self.observations = dict()

        # pause on initial start
        self.speed = 0.0
        self.get_logger().info("Dreamer node has started")

        # variables
        self.time = self.get_clock().now()
        self.scan_num = 0

        # subscribers/publishers
        self.sub_scan = self.create_subscription(
            LaserScan, self.const.LIDAR_TOPIC, self.scan_callback, 10
        )

        self.pub_drive = self.create_publisher(
            AckermannDriveStamped, self.const.DRIVE_TOPIC, 10
        )

        self.pub_scan = self.create_publisher(LaserScan, self.const.LIDAR_TOPIC, 10)

        self.dreamer_init(self.config)

    def dreamer_init(self, config: Config):
        tools.set_seed_everywhere(config.SEED)
        if config.DETERMINISTIC_RUN:
            tools.enable_deterministic_run()
        logdir = pathlib.Path(config.LOG_DIRECTORY).expanduser()
        config.TRAINING_DIRECTORY = config.TRAINING_DIRECTORY or logdir / "train_eps"
        config.EVALUATION_DIRECTORY = config.EVALUATION_DIRECTORY or logdir / "eval_eps"
        config.STEPS //= config.ACTION_REPEAT
        config.EVALUATION_EVERY //= config.ACTION_REPEAT
        config.LOG_EVERY //= config.ACTION_REPEAT
        config.TIME_LIMIT //= config.ACTION_REPEAT

        print("Logdir", logdir)
        logdir.mkdir(parents=True, exist_ok=True)
        config.TRAINING_DIRECTORY.mkdir(parents=True, exist_ok=True)
        config.EVALUATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
        step = sum(
            int(str(n).split("-")[-1][:-4]) - 1
            for n in config.TRAINING_DIRECTORY.glob("*.npz")
        )
        # step in logger is environmental step
        logger = tools.Logger(logdir, config.ACTION_REPEAT * step)

        print("Create envs.")
        if config.OFFLINE_TRAINING_DIRECTORY:
            directory = config.OFFLINE_TRAINING_DIRECTORY.format(**vars(config))
        else:
            directory = config.TRAINING_DIRECTORY
        train_eps = tools.load_episodes(directory, limit=config.DATASET_SIZE)
        if config.OFFLINE_EVALUATION_DIRECTORY:
            directory = config.OFFLINE_EVALUATION_DIRECTORY.format(**vars(config))
        else:
            directory = config.EVALUATION_DIRECTORY
        eval_eps = tools.load_episodes(directory, limit=1)

        def make(mode, id):
            return self.make_env(config, mode, id)

        train_envs = [make("train", i) for i in range(config.ENVIRONMENT_COUNT)]
        eval_envs = [make("eval", i) for i in range(config.ENVIRONMENT_COUNT)]

        if config.PARALLEL:
            train_envs = [Parallel(env, "process") for env in train_envs]
            eval_envs = [Parallel(env, "process") for env in eval_envs]
        else:
            train_envs = [Damy(env) for env in train_envs]
            eval_envs = [Damy(env) for env in eval_envs]

        acts = train_envs[0].action_space
        print("Action Space", acts)
        config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

        if not config.OFFLINE_TRAINING_DIRECTORY:
            prefill = max(
                0,
                config.PREFILL
                - sum(
                    int(str(n).split("-")[-1][:-4]) - 1
                    for n in config.TRAINING_DIRECTORY.glob("*.npz")
                ),
            )
            print(f"Prefill dataset ({prefill} steps).")
            if hasattr(acts, "discrete"):
                random_actor = tools.OneHotDist(
                    torch.zeros(config.num_actions).repeat(config.ENVIRONMENT_COUNT, 1)
                )
            else:
                random_actor = torchd.independent.Independent(
                    torchd.uniform.Uniform(
                        torch.tensor(acts.low).repeat(config.ENVIRONMENT_COUNT, 1),
                        torch.tensor(acts.high).repeat(config.ENVIRONMENT_COUNT, 1),
                    ),
                    1,
                )

            def random_agent(o, d, s):
                action = random_actor.sample()
                logprob = random_actor.log_prob(action)
                return {"action": action, "logprob": logprob}, None

            tools.simulate(
                random_agent,
                train_envs,
                train_eps,
                config.TRAINING_DIRECTORY,
                logger,
                limit=config.DATASET_SIZE,
                steps=prefill,
            )
            logger.step += prefill * config.ACTION_REPEAT
            print(f"Logger: ({logger.step} steps).")

        print("Simulate agent.")
        train_dataset = self.make_dataset(train_eps, config)
        self.make_dataset(eval_eps, config)
        agent = Dreamer(
            train_envs[0].observation_space,
            train_envs[0].action_space,
            config,
            logger,
            train_dataset,
        ).to(config.DEVICE)
        agent.requires_grad_(requires_grad=False)
        if (logdir / "latest.pt").exists():
            checkpoint = torch.load(logdir / "latest.pt")
            agent.load_state_dict(checkpoint["agent_state_dict"])
            tools.recursively_load_optim_state_dict(
                agent, checkpoint["optims_state_dict"]
            )
            agent._should_pretrain._once = False

    def make_env(self, config: Config, mode, id):
        # port the dreamer environment
        pass

    def make_dataset(self, episodes, config: Config):
        generator = tools.sample_episodes(episodes, config.BATCH_LENGTH)
        dataset = tools.from_generator(generator, config.BATCH_SIZE)
        return dataset

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

        steering = 0.0
        speed = 1.0

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

        if duration.nanoseconds / 1e9 < (0.08 - 0.001):  # limit to approx. 10Hz
            assert f"Error: last laser scan time is {last_scan_time}. Limit = 10Hz"
            return

        # receive LiDAR scan rte
        self.time = current_stamp

        filtered_data = self._filter_range(lidar_data, self.const.FORWARD_SCAN_ARC)
        filtered_data = [round(point_range, 4) for point_range in filtered_data]

        if not hasattr(self, "last_log_time"):
            self.last_log_time = self.get_clock().now()

        elapsed_time = (self.get_clock().now() - self.last_log_time).nanoseconds / 1e9

        if elapsed_time >= 3:
            self.last_log_time = self.get_clock().now()
            print(
                "dreamer received LIDAR scan @ rate:", last_scan_time / 1e9
            )  # TODO: debug for scan rate
            print("observation = ", filtered_data)  # TODO: debug range

        obs_lidar = filtered_data
        return obs_lidar

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
        print(
            "dreamer published action: steering_angle = ",
            steering_angle,
            "; speed = ",
            speed,
        )
        return drive_msg

    def recursive_update(self, base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                self.recursive_update(base[key], value)
            else:
                base[key] = value

    def _filter_range(self, lidar_data: LaserScan, range: Tuple[float]) -> List[float]:
        """
        Filters raw scan for only the desired arc.

        Args:
            lidar_data: original scan message
            range: the arc to filter for (e.g. (-pi/2, +pi/2))

        Returns:
            List of filtered ranges
        """
        raw_scan = list(
            np.flip(np.array(lidar_data.ranges, dtype=float))
        )  # include liDAR observations

        raw_min_angle = lidar_data.angle_min
        raw_max_angle = lidar_data.angle_max
        target_min_angle, target_max_angle = range

        raw_arc_length = raw_max_angle - raw_min_angle
        number_of_ranges = len(raw_scan)
        number_of_ranges_per_radian = number_of_ranges / raw_arc_length

        skipped_min_angle = target_min_angle - raw_min_angle
        skipped_max_angle = raw_max_angle - target_max_angle

        min_index = math.ceil(skipped_min_angle * number_of_ranges_per_radian)
        max_index = math.floor(
            number_of_ranges - skipped_max_angle * number_of_ranges_per_radian
        )

        return raw_scan[min_index:max_index]


def main(args=None):
    rclpy.init(args=args)
    print("Dreamer Initialized")
    dreamer_node = DreamerRacer()
    rclpy.spin(dreamer_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    dreamer_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
