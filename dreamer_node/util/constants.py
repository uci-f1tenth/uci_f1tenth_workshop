# constants.py
from pathlib import Path
import numpy as np


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

        # Dreamer
        self.DEVICE = "cuda"

        # action space
        self.min_steering = -0.418
        self.max_steering = 0.418
        self.min_speed = 1.5
        self.max_speed = 19.67


class Config:
    def __init__(self):
        self.logdir = Path("./logdir/f1tenth")
        self.traindir = None#Path("/dreamer/resource/train")
        self.evaldir = None#Path("/dreamer/resource/eval")
        self.offline_traindir = ''#"/dreamer/resource/train"
        self.offline_evaldir = ''#"/dreamer/resource/eval"
        self.seed = 0
        self.deterministic_run = False
        self.steps = int(1e6)  # Updated from f1tenth
        self.parallel = False
        self.eval_every = int(1e4)
        self.eval_episode_num = 10
        self.log_every = int(1e4)
        self.reset_every = 0
        self.device = "cuda:0"
        self.compile = True
        self.precision = 32
        self.debug = False
        self.video_pred_log = True
        self.task = "custom_task"  # Updated from f1tenth
        self.envs = 1  # Updated from f1tenth
        self.action_repeat = 1  # Updated from f1tenth
        self.time_limit = 1000
        self.grayscale = False
        self.prefill = 2500
        self.reward_EMA = True
        self.dyn_hidden = 512
        self.dyn_deter = 512
        self.dyn_stoch = 32
        self.dyn_discrete = 32
        self.dyn_rec_depth = 1
        self.dyn_mean_act = "none"
        self.dyn_std_act = "sigmoid2"
        self.dyn_min_std = 0.1
        self.grad_heads = ["decoder", "reward", "cont"]
        self.units = 512
        self.act = "SiLU"
        self.norm = True

        # Encoder configuration (updated from f1tenth)
        self.encoder = {
            "mlp_keys": "pose|velocity|acceleration|lidar",  # Process these with MLP
            "cnn_keys": "image",  # Process this with CNN
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
        self.decoder = {
            "mlp_keys": "pose|velocity|acceleration|lidar",  # Decode these with MLP
            "cnn_keys": "image",  # Decode this with CNN
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
        self.actor = {
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
        self.critic = {
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
        self.reward_head = {
            "layers": 2,
            "dist": "symlog_disc",
            "loss_scale": 1.0,
            "outscale": 0.0,
        }

        # Continuation head configuration
        self.cont_head = {
            "layers": 2,
            "loss_scale": 1.0,
            "outscale": 1.0,
        }

        # Dynamics and representation scaling
        self.dyn_scale = 0.5
        self.rep_scale = 0.1
        self.kl_free = 1.0
        self.weight_decay = 0.0
        self.unimix_ratio = 0.01
        self.initial = "learned"
        self.batch_size = 16
        self.batch_length = 64
        self.train_ratio = 512  # Updated from f1tenth
        self.pretrain = 100
        self.model_lr = 1e-4
        self.opt_eps = 1e-8
        self.grad_clip = 1000
        self.dataset_size = 1000000
        self.opt = "adam"
        self.discount = 0.997
        self.discount_lambda = 0.95
        self.imag_horizon = 15
        self.imag_gradient = "dynamics"
        self.imag_gradient_mix = 0.0
        self.eval_state_mean = False
        self.expl_behavior = "greedy"
        self.expl_until = 0
        self.expl_extr_scale = 0.0
        self.expl_intr_scale = 1.0
        self.disag_target = "stoch"
        self.disag_log = True
        self.disag_models = 10
        self.disag_offset = 1
        self.disag_layers = 4
        self.disag_units = 400
        self.disag_action_cond = False