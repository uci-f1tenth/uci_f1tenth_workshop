from pathlib import Path


class Config:
    logdir = Path("./logdir/f1tenth")
    traindir: Path | None = None  # Path("/dreamer/resource/train")
    evaldir: Path | None = None  # Path("/dreamer/resource/eval")
    offline_traindir = ""  # "/dreamer/resource/train"
    offline_evaldir = ""  # "/dreamer/resource/eval"
    seed = 0
    deterministic_run = False
    steps = int(1e6)  # Updated from f1tenth
    parallel = False
    eval_every = int(1e4)
    eval_episode_num = 1
    log_every = int(1e4)
    reset_every = 0
    device = "cuda:0"
    compile = True
    precision = 32
    debug = False
    video_pred_log = True
    task = "custom_task"  # Updated from f1tenth
    envs = 1  # Updated from f1tenth
    action_repeat = 1  # Updated from f1tenth
    time_limit = 1000
    grayscale = False
    prefill = 2500
    reward_EMA = True
    dyn_hidden = 512
    dyn_deter = 512
    dyn_stoch = 32
    dyn_discrete = 32
    dyn_rec_depth = 1
    dyn_mean_act = "none"
    dyn_std_act = "sigmoid2"
    dyn_min_std = 0.1
    grad_heads = ["decoder", "reward", "cont"]
    units = 512
    act = "SiLU"
    norm = True

    # Encoder configuration (updated from f1tenth)
    encoder = {
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
    decoder = {
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
    actor = {
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
    critic = {
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
    reward_head = {
        "layers": 2,
        "dist": "symlog_disc",
        "loss_scale": 1.0,
        "outscale": 0.0,
    }

    # Continuation head configuration
    cont_head = {
        "layers": 2,
        "loss_scale": 1.0,
        "outscale": 1.0,
    }

    # Dynamics and representation scaling
    dyn_scale = 0.5
    rep_scale = 0.1
    kl_free = 1.0
    weight_decay = 0.0
    unimix_ratio = 0.01
    initial = "learned"
    batch_size = 16
    batch_length = 64
    train_ratio = 512  # Updated from f1tenth
    pretrain = 100
    model_lr = 1e-4
    opt_eps = 1e-8
    grad_clip = 1000
    dataset_size = 1000000
    opt = "adam"
    discount = 0.997
    discount_lambda = 0.95
    imag_horizon = 15
    imag_gradient = "dynamics"
    imag_gradient_mix = 0.0
    eval_state_mean = False
    expl_behavior = "greedy"
    expl_until = 0
    expl_extr_scale = 0.0
    expl_intr_scale = 1.0
    disag_target = "stoch"
    disag_log = True
    disag_models = 10
    disag_offset = 1
    disag_layers = 4
    disag_units = 400
    disag_action_cond = False
