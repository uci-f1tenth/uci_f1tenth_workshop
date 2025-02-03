# constants.py
import numpy as np
class Constants:
    """
    Class to hold constants for the car.
    """
    def __init__(self):

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
        self.DEVICE = "cuda"
        self.DREAMER_LOGDIR = "./logdir"
        self.DREAMER_TRAINDIR = None
        self.DREAMER_EVALDIR = None
        self.DREAMER_OFFLINE_TRAINDIR = ''
        self.DREAMER_OFFLINE_EVALDIR = ''
        self.DREAMER_SEED = 0
        self.DREAMER_DETERMINISTIC_RUN = False
        self.DREAMER_STEPS = 1e6
        self.DREAMER_PARALLEL = False
        self.DREAMER_EVAL_EVERY = 1e4
        self.DREAMER_EVAL_EPISODE_NUM = 10
        self.DREAMER_LOG_EVERY = 1e4
        self.DREAMER_RESET_EVERY = 0
        self.DREAMER_DEVICE = 'cuda:0'  # Or 'cpu'
        self.DREAMER_COMPILE = True
        self.DREAMER_PRECISION = 32
        self.DREAMER_DEBUG = False
        self.DREAMER_VIDEO_PRED_LOG = True

        # Environment
        self.DREAMER_TASK = 'f1tenth'
        self.DREAMER_SIZE = [64, 64]
        self.DREAMER_ENVS = 1
        self.DREAMER_ACTION_REPEAT = 2
        self.DREAMER_TIME_LIMIT = 1000
        self.DREAMER_GRAYSCALE = False
        self.DREAMER_PREFILL = 2500

        # Model
        self.DREAMER_DYN_HIDDEN = 512
        self.DREAMER_DYN_DETER = 512
        self.DREAMER_DYN_STOCH = 32
        self.DREAMER_UNITS = 512
        self.DREAMER_ENCODER = {
            'mlp_keys': '$^', 'cnn_keys': 'image', 'act': 'SiLU', 'norm': True,
            'cnn_depth': 32, 'kernel_size': 4, 'minres': 4, 'mlp_layers': 5,
            'mlp_units': 1024, 'symlog_inputs': True
        }
        self.DREAMER_DECODER = {
            'mlp_keys': '$^', 'cnn_keys': 'image', 'act': 'SiLU', 'norm': True,
            'cnn_depth': 32, 'kernel_size': 4, 'minres': 4, 'mlp_layers': 5,
            'mlp_units': 1024, 'cnn_sigmoid': False, 'image_dist': 'mse',
            'vector_dist': 'symlog_mse', 'outscale': 1.0
        }
        self.DREAMER_ACTOR = {
            'layers': 2, 'dist': 'tanh_normal', 'entropy': 3e-4,
            'unimix_ratio': 0.01, 'std': 'learned', 'min_std': 0.1,
            'max_std': 1.0, 'temp': 0.1, 'lr': 3e-5, 'eps': 1e-5,
            'grad_clip': 100.0, 'outscale': 1.0
        }
        self.DREAMER_CRITIC = {
            'layers': 2, 'dist': 'symlog_disc', 'slow_target': True,
            'slow_target_update': 1, 'slow_target_fraction': 0.02, 'lr': 3e-5,
            'eps': 1e-5, 'grad_clip': 100.0, 'outscale': 0.0
        }
        self.DREAMER_REWARD_HEAD = {
            'layers': 2, 'dist': 'symlog_disc', 'loss_scale': 1.0,
            'outscale': 0.0
        }
        self.DREAMER_CONT_HEAD = {
            'layers': 2, 'loss_scale': 1.0, 'outscale': 1.0
        }
        self.DREAMER_DYN_SCALE = 0.5
        self.DREAMER_REP_SCALE = 0.1
        self.DREAMER_KL_FREE = 1.0
        self.DREAMER_WEIGHT_DECAY = 0.0
        self.DREAMER_UNIMIX_RATIO = 0.01
        self.DREAMER_INITIAL = 'learned'

        # Training
        self.DREAMER_BATCH_SIZE = 16
        self.DREAMER_BATCH_LENGTH = 64
        self.DREAMER_TRAIN_RATIO = 512
        self.DREAMER_PRETRAIN = 100
        self.DREAMER_MODEL_LR = 1e-4
        self.DREAMER_OPT_EPS = 1e-8
        self.DREAMER_GRAD_CLIP = 1000
        self.DREAMER_DATASET_SIZE = 1000000
        self.DREAMER_OPT = 'adam'

        # Behavior
        self.DREAMER_DISCOUNT = 0.997
        self.DREAMER_DISCOUNT_LAMBDA = 0.95
        self.DREAMER_IMAG_HORIZON = 15
        self.DREAMER_IMAG_GRADIENT = 'dynamics'
        self.DREAMER_IMAG_GRADIENT_MIX = 0.0
        self.DREAMER_EVAL_STATE_MEAN = False

        # Exploration
        self.DREAMER_EXPL_BEHAVIOR = 'greedy'
        self.DREAMER_EXPL_UNTIL = 0
        self.DREAMER_EXPL_EXTR_SCALE = 0.0
        self.DREAMER_EXPL_INTR_SCALE = 1.0
        self.DREAMER_DISAG_TARGET = 'stoch'
        self.DREAMER_DISAG_LOG = True
        self.DREAMER_DISAG_MODELS = 10
        self.DREAMER_DISAG_OFFSET = 1
        self.DREAMER_DISAG_LAYERS = 4
        self.DREAMER_DISAG_UNITS = 400
        self.DREAMER_DISAG_ACTION_COND = False
