#!/usr/bin/env python
from __future__ import print_function
import sys
import math
import numpy as np

#ROS Imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

#PID CONTROL PARAMS
kp = 1.0
kd = 0.001
ki = 0.005
servo_offset = 0.0
prev_error = 0.0 
error = 0.0
integral = 0.0
prev_time = None

#WALL FOLLOW PARAMS
ANGLE_RANGE = 270 # Hokuyo 10LX has 270 degrees scan
DESIRED_DISTANCE_RIGHT = 0.9 # meters
DESIRED_DISTANCE_LEFT = 0.85
VELOCITY = 1.5 # meters per second
CAR_LENGTH = 1.0 # Traxxas Rally is 20 inches or 0.5 meters

class WallFollow(Node):
    """ Implement Wall Following on the car
    """
    def __init__(self):
        global prev_time
#        super.__init__('wall_follow_node')

        #Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'
        prev_time = self.get_clock().now()

        self.lidar_sub = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

    def getRange(self, data, angle):
        # data: single message from topic /scan
        # angle: between -45 to 225 degrees, where 0 degrees is directly to the right
        # Outputs length in meters to object with angle in lidar scan field of view
        #make sure to take care of nans etc.
        #TODO: implement
        if angle >= -45 and angle <= 225:
            iterator = len(data) * (angle + 90) / 360
            if not np.isnan(data[int(iterator)]) and not np.isinf(data[int(iterator)]):
                return data[int(iterator)]

    def pid_control(self, error, velocity):
        global integral
        global prev_error
        global kp
        global ki
        global kd
        global prev_time
        angle = 0.0
        current_time = self.get_clock().now()
        del_time = current_time - prev_time
        #TODO: Use kp, ki & kd to implement a PID controller for 
        integral += prev_error * del_time
        angle = kp * error + ki * integral + kd * (error - prev_error) / del_time
        prev_error = error
        prev_time = current_time
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "laser"
        drive_msg.drive.steering_angle = -angle
        if abs(angle) > math.radians(0) and abs(angle) <= math.radians(10):
            drive_msg.drive.speed = velocity
        elif abs(angle) > math.radians(10) and abs (angle) <= math.radians(20):
            drive_msg.drive.speed = 1.0
        else:
            drive_msg.drive.speed = 0.5
        self.drive_pub.publish(drive_msg)

    def followLeft(self, data, leftDist):
        #Follow left wall as per the algorithm 
        #TODO:implement
        front_scan_angle = 125
        back_scan_angle = 180
        teta = math.radians(abs(front_scan_angle - back_scan_angle))
        front_scan_dist = self.getRange(data, front_scan_angle)
        back_scan_dist = self.getRange(data, back_scan_angle)
        alpha = math.atan2(front_scan_dist * math.cos(teta) - back_scan_dist, front_scan_dist * math.sin(teta))
        wall_dist = back_scan_dist * math.cos(alpha)
        ahead_wall_dist = wall_dist + CAR_LENGTH * math.sin(alpha)
        return leftDist - ahead_wall_dist

    def lidar_callback(self, data):
        """ 
        """
        error = self.followLeft(data.ranges, DESIRED_DISTANCE_LEFT) #TODO: replace with error returned by followLeft
        #send error to pid_control
        self.pid_control(error, VELOCITY)

def main(args=None):
    rclpy.init()
    wall_follow_node = WallFollow()
    rclpy.spin(wall_follow_node)
    wall_follow_node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
	main()
