# constants.py
import numpy as np
class Constants:
    """
    Class to hold constants for the car.
    """
    def __init__(self):

        # numeric variables
        self.MAX_SPEED = 7.0 # meters/second
        self.MAX_DECEL = 8.26 # meters/second^2
        self.MAX_STOP_DISTANCE = np.square(self.MAX_SPEED) / (2.0 * self.MAX_DECEL)
        self.LOOKAHEAD_DISTANCE = 2.0 * self.MAX_STOP_DISTANCE
        self.VEHICLE_WIDTH = (0.3302 * 1.2) # use 120% of the wheelbase as vehicle width

        self.DT_THRES = 1.0 / 50.0 # run computation at max. 50 Hz

        self.FORWARD_SCAN_ARC = (np.deg2rad(-90.0), np.deg2rad(+90.0))
        self.HEAD_COMPUTATION_ARC = np.deg2rad(30.0)
        self.HEAD_COMPUTATION_PERCENTILE = 100 * (1.0 - (self.HEAD_COMPUTATION_ARC / \
                                                            (self.FORWARD_SCAN_ARC[1] - self.FORWARD_SCAN_ARC[0])))
        
        self.MIN_GAP_LENGTH = 0.2 # meters
        self.MED_RANGE_DEVIATION_THRES = 9.0  # outlier test: x[i] > median(x) * median_deviation_threshold

        self.HEAD_ERR = 0.0

        self.STEER_ANGLE = 0.0
        self.VEHICLE_SPEED = 0.0
        self.MAX_VEHICLE_SPEED = 6.0

        self.MAX_STEER_ANGLE = np.deg2rad(24) # maximum (absolute) steering angle

        # PID Gains
        # TODO: tune PID
        self.KP = 1.0
        self.KD = 0.0
        self.KI = 0.1

        # Topics
        self.LIDAR_TOPIC = '/scan'
        self.DRIVE_TOPIC = '/drive'
        self.ODOMETRY_TOPIC = '/ego_racecar/odom'
