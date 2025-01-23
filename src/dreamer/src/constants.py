# constants.py

class Constants:
    """
    Class to hold constants for the car.
    """
    def __init__(self):
        # TODO
        self.SPEED_LIMIT = 2.0

        # PID Gains
        # TODO: tune PID
        self.KP = 0.1
        self.KD = 0.0
        self.KI = 0.0

        # Topics
        self.LIDAR_TOPIC = '/scan'
        self.DRIVE_TOPIC = '/drive'
        self.ODOMETRY_TOPIC = '/ego_racecar/odom'
