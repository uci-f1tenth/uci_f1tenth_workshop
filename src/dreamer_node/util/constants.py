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
        self.LIDAR_TOPIC = '/scan'
        self.DRIVE_TOPIC = '/drive'
        self.ODOMETRY_TOPIC = '/ego_racecar/odom'

        # Dreamer
        self.DEVICE = "cuda"

class Config:
    def __init__(self):
        self.logdir = Path("/dreamer/resource/log")
        self.traindir = Path("/dreamer/resource/train")
        self.evaldir = Path("/dreamer/resource/eval")
        self.offline_traindir = "/dreamer/resource/train"
        self.offline_evaldir = "/dreamer/resource/eval"
        self.seed = 0
        self.deterministic_run = False
        self.steps = 1e6
        self.parallel = False
        self.eval_every = 1e4
        self.eval_episode_num = 10
        self.log_every = 1e4
        self.reset_every = 0
        self.device = "cuda:0"
        self.compile = True
        self.precision = 32
        self.debug = False
        self.video_pred_log = True
        self.task = "f1tenth"
        # self.size = [64, 64]
        self.envs = 1
        self.action_repeat = 2
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
        self.encoder = {  # ... (encoder config)
            "mlp_keys": "$^",
            "cnn_keys": "image",
            "act": "SiLU",
            "norm": True,
            "cnn_depth": 32,
            "kernel_size": 4,
            "minres": 4,
            "mlp_layers": 5,
            "mlp_units": 1024,
            "symlog_inputs": True,
        }
        self.decoder = {  # ... (decoder config)
            "mlp_keys": "$^",
            "cnn_keys": "image",
            "act": "SiLU",
            "norm": True,
            "cnn_depth": 32,
            "kernel_size": 4,
            "minres": 4,
            "mlp_layers": 5,
            "mlp_units": 1024,
            "cnn_sigmoid": False,
            "image_dist": "mse",
            "vector_dist": "symlog_mse",
            "outscale": 1.0,
        }
        self.actor = {  # ... (actor config)
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
        self.critic = { # ... (critic config)
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

        self.reward_head = { # ... (reward_head config)
            "layers": 2,
            "dist": "symlog_disc",
            "loss_scale": 1.0,
            "outscale": 0.0,
        }
        self.cont_head = { # ... (cont_head config)
            "layers": 2,
            "loss_scale": 1.0,
            "outscale": 1.0,
        }
        self.dyn_scale = 0.5
        self.rep_scale = 0.1
        self.kl_free = 1.0
        self.weight_decay = 0.0
        self.unimix_ratio = 0.01
        self.initial = "learned"
        self.batch_size = 16
        self.batch_length = 64
        self.train_ratio = 512
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

        self.f1tenth = {
            "steps": 1e6,
            "action_repeat": 1,
            "envs": 1,
            "train_ratio": 512,
            "encoder": {"mlp_keys": "lidar", "cnn_keys": "image", "mlp_units":512,
                        "act": "SiLU", "cnn_depth": 32, "mlp_layers":4, "symlog_inpts":False,
                        "norm": True, "kernel_size":4, "minres":4},
            "decoder": {"mlp_keys": "lidar", "cnn_keys": "image", "norm": True,
                        "act": "SiLU", "mlp_units":512,"cnn_depth": 32, "mlp_layers":4,
                        "kernel_size":4,"minres":4, "cnn_sigmoid": True, "image_dist": "mse", 
                        "vector_dist": "normal", "outscale": 1.0},
        }
