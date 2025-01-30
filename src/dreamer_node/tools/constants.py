# constants.py
import numpy as np
class Constants:
    """
    Class to hold constants for the car.
    """
    def __init__(self):
        # Vehicle Stats
        self.MAX_VEHICLE_SPEED = 6.0 # meters/second
        self.VEHICLE_WIDTH = (0.3302 * 1.2) # use 120% of the wheelbase as vehicle width
        self.MAX_STOP_DISTANCE = np.square(self.MAX_SPEED) / (2.0 * self.MAX_DECEL)
        self.LOOKAHEAD_DISTANCE = 2.0 * self.MAX_STOP_DISTANCE

        # Constraints
        self.MAX_SPEED = 7.0 # meters/second
        self.MAX_DECEL = 8.26 # meters/second^2
        self.MAX_STEER_ANGLE = np.deg2rad(24) # maximum (absolute) steering angle        

        # Sensors
        ## Lidar
        self.FORWARD_SCAN_ARC = (np.deg2rad(-90.0), np.deg2rad(+90.0))
        self.HEAD_COMPUTATION_ARC = np.deg2rad(30.0)
        self.HEAD_COMPUTATION_PERCENTILE = 100 * (1.0 - (self.HEAD_COMPUTATION_ARC / \
                                            (self.FORWARD_SCAN_ARC[1] - self.FORWARD_SCAN_ARC[0])))
        self.MIN_GAP_LENGTH = 0.2 # meters
        self.MED_RANGE_DEVIATION_THRES = 9.0  # outlier test: x[i] > median(x) * median_deviation_threshold

        # Computations
        self.DT_THRES = 1.0 / 50.0 # run computation at max. 50 Hz

        # PID Gains
        # TODO: tune PID
        self.KP = 1.0
        self.KD = 0.0
        self.KI = 0.1

        # Topics
        self.LIDAR_TOPIC = '/scan'
        self.DRIVE_TOPIC = '/drive'
        self.ODOMETRY_TOPIC = '/ego_racecar/odom'

        # Dreamer
        ## General
        self.STEPS = 5e6  # Total number of training steps
        self.EVAL_EVERY = 1e4  # Frequency of evaluations (in steps)
        self.LOG_EVERY = 1e3  # Frequency of logging (in steps)
        self.LOG_SCALARS = True  # Whether to log scalar values
        self.LOG_IMAGES = True  # Whether to log image-based observations
        self.LOG_VIDEOS = True  # Whether to log videos of episodes
        self.GPU_GROWTH = True  # Enable dynamic GPU memory allocation
        self.PRECISION = 32  # Floating point precision (32-bit)

        ## Environment
        self.TRACK = 'austria'  # Track used for racing simulation
        self.TASK = 'max_progress'  # Task type (e.g., maximizing progress)
        self.ACTION_REPEAT = 4  # Number of times an action is repeated before re-evaluating
        self.EVAL_EPISODES = 5  # Number of episodes used for evaluation
        self.TIME_LIMIT_TRAIN = 2000  # Maximum steps per training episode
        self.TIME_LIMIT_TEST = 4000  # Maximum steps per test episode
        self.PREFILL_AGENT = 'gap_follower'  # Agent used for pre-filling experience buffer
        self.PREFILL = 5000  # Number of initial random steps before training
        self.EVAL_NOISE = 0.0  # Noise level added during evaluation (0 = deterministic)
        self.CLIP_REWARDS = 'none'  # Whether to clip rewards ('none' = no clipping)
        self.CLIP_REWARDS_MIN = -1  # Minimum reward value (if clipping enabled)
        self.CLIP_REWARDS_MAX = 1  # Maximum reward value (if clipping enabled)

        ## Model
        self.ENCODED_OBS_DIM = 1080  # Dimensionality of the encoded lidar observation
        self.DETER_SIZE = 200  # Size of deterministic latent state in RSSM
        self.STOCH_SIZE = 30  # Size of stochastic latent state in RSSM
        self.NUM_UNITS = 400  # Number of hidden units per layer in networks
        self.REWARD_OUT_DIST = 'normal'  # Distribution used for reward prediction
        self.DENSE_ACT = 'elu'  # Activation function for dense layers
        self.CNN_ACT = 'relu'  # Activation function for convolutional layers
        self.PCONT = True  # Whether to use predictive continuation (future prediction)
        self.FREE_NATS = 3.0  # Free nats for KL divergence loss (prevents collapse)
        self.KL_SCALE = 1.0  # Scaling factor for KL loss in RSSM
        self.PCONT_SCALE = 10.0  # Scaling factor for predictive continuation loss
        self.WEIGHT_DECAY = 0.0  # Weight decay regularization factor
        self.WEIGHT_DECAY_PATTERN = r'.*'  # Pattern for applying weight decay

        ## Training
        self.BATCH_SIZE = 50  # Number of sequences per training batch
        self.BATCH_LENGTH = 50  # Length of each training sequence
        self.TRAIN_EVERY = 1000  # Number of steps between training updates
        self.TRAIN_STEPS = 100  # Number of training steps per update
        self.PRETRAIN = 100  # Number of pretraining steps before RL training
        self.MODEL_LR = 6e-4  # Learning rate for the model (RSSM)
        self.VALUE_LR = 8e-5  # Learning rate for the value function
        self.ACTOR_LR = 8e-5  # Learning rate for the policy (actor network)
        self.GRAD_CLIP = 1.0  # Gradient clipping threshold
        self.DATASET_BALANCE = False  # Whether to balance dataset during training

        ## Behavior
        self.DISCOUNT = 0.99  # Discount factor for future rewards (gamma)
        self.DISCLAM = 0.95  # Lambda parameter for TD-lambda returns
        self.HORIZON = 15  # Planning horizon for model-based predictions
        self.ACTION_DIST = 'tanh_normal'  # Distribution type for action sampling
        self.ACTION_INIT_STD = 5.0  # Initial standard deviation for action distribution
        self.EXPL = 'additive_gaussian'  # Type of exploration noise
        self.EXPL_AMOUNT = 0.3  # Initial amount of exploration noise
        self.EXPL_DECAY = 0.0  # Rate at which exploration noise decays
        self.EXPL_MIN = 0.3  # Minimum exploration noise level