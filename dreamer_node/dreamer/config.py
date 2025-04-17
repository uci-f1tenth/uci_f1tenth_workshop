from pathlib import Path


class Config:
    LOG_DIRECTORY = Path("./logdir/f1tenth")
    TRAINING_DIRECTORY: Path | None = None  # Path("/dreamer/resource/train")
    EVALUATION_DIRECTORY: Path | None = None  # Path("/dreamer/resource/eval")
    OFFLINE_TRAINING_DIRECTORY = ""  # "/dreamer/resource/train"
    OFFLINE_EVALUATION_DIRECTORY = ""  # "/dreamer/resource/eval"
    SEED = 0
    DETERMINISTIC_RUN = False
    STEPS = int(1e6)  # Updated from f1tenth
    PARALLEL = False
    EVALUATION_EVERY = int(1e4)
    EVALUATION_EPISODE_NUMBER = 1
    LOG_EVERY = int(1e4)
    RESET_EVERY = 0
    DEVICE = "cuda:0"
    COMPILE = True
    PRECISION = 32
    DEBUG = False
    VIDEO_PREDICTION_LOG = True
    TASK = "custom_task"  # Updated from f1tenth
    ENVIRONMENT_COUNT = 1  # Updated from f1tenth
    ACTION_REPEAT = 1  # Updated from f1tenth
    TIME_LIMIT = 1000
    GRAYSCALE = False
    PREFILL = 2500
    REWARD_EMA = True
    DYNAMIC_HIDDEN = 512
    DYNAMIC_DETER = 512
    DYNAMIC_STOCH = 32
    DYNAMIC_DISCRETE = 32
    DYNAMIC_REC_DEPTH = 1
    DYNAMIC_MEAN_ACT = "none"
    DYNAMIC_STD_ACT = "sigmoid2"
    DYNAMIC_MIN_STD = 0.1
    GRAD_HEADS = ["decoder", "reward", "cont"]
    UNITS = 512
    ACT = "SiLU"
    NORM = True

    # Encoder configuration (updated from f1tenth)
    ENCODER = {
        "mlp_keys": "lidar",  # Process these with MLP
        "cnn_keys": "$^",  # Process this with CNN
        "act": "SiLU",
        "norm": True,
        "cnn_depth": 32,
        "kernel_size": 4,
        "minres": 4,
        "mlp_layers": 4,
        "mlp_units": 512,
        "symlog_inputs": False,
    }

    # Decoder configuration (updated from f1tenth)
    DECODER = {
        "mlp_keys": "lidar",  # Decode these with MLP
        "cnn_keys": "$^",  # Decode this with CNN
        "act": "SiLU",
        "norm": True,
        "cnn_depth": 32,
        "kernel_size": 4,
        "minres": 4,
        "mlp_layers": 4,
        "mlp_units": 512,
        "cnn_sigmoid": True,
        "image_dist": "mse",
        "vector_dist": "normal",
        "outscale": 1.0,
    }

    # Actor configuration
    ACTOR = {
        "layers": 2,
        "dist": "normal",
        "entropy": 3e-4,
        "unimix_ratio": 0.01,
        "std": "learned",
        "min_std": 0.1,
        "max_std": 1.0,
        "temp": 0.1,
        "lr": 3e-5,
        "eps": 1e-5,
        "grad_clip": 100.0,
        "outscale": 1.0,
    }

    # Critic configuration
    CRITIC = {
        "layers": 2,
        "dist": "symlog_disc",
        "slow_target": True,
        "slow_target_update": 1,
        "slow_target_fraction": 0.02,
        "lr": 3e-5,
        "eps": 1e-5,
        "grad_clip": 100.0,
        "outscale": 0.0,
    }

    # Reward head configuration
    REWARD_HEAD = {
        "layers": 2,
        "dist": "symlog_disc",
        "loss_scale": 1.0,
        "outscale": 0.0,
    }

    # Continuation head configuration
    CONTINUATION_HEAD = {
        "layers": 2,
        "loss_scale": 1.0,
        "outscale": 1.0,
    }

    # Dynamics and representation scaling
    DYNAMIC_SCALE = 0.5
    REP_SCALE = 0.1
    KL_FREE = 1.0
    WEIGHT_DECAY = 0.0
    UNIMIX_RATIO = 0.01
    INITIAL = "learned"
    BATCH_SIZE = 16
    BATCH_LENGTH = 64
    TRAIN_RATIO = 512  # Updated from f1tenth
    PRETRAIN = 100
    MODEL_LR = 1e-4
    OPT_EPS = 1e-8
    GRAD_CLIP = 1000
    DATASET_SIZE = 1000000
    OPT = "adam"
    DISCOUNT = 0.997
    DISCOUNT_LAMBDA = 0.95
    IMAGE_HORIZON = 15
    IMAGE_GRADIENT = "dynamics"
    IMAGE_GRADIENT_MIX = 0.0
    EVALUATION_STATE_MEAN = False
    EXPLORATION_BEHAVIOR = "greedy"
    EXPLORATION_UNTIL = 0
    EXPLORATION_EXTRA_SCALE = 0.0
    EXPLORATION_INTRO_SCALE = 1.0
    DISAGREE_TARGET = "stoch"
    DISAGREE_LOG = True
    DISAGREE_MODELS = 10
    DISAGREE_OFFSET = 1
    DISAGREE_LAYERS = 4
    DISAGREE_UNITS = 400
    DISAGREE_ACTION_CONDITION = False
