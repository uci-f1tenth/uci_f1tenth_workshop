import argparse
import functools
import os
import pathlib
import sys

import numpy as np

import gymnasium
import racecar_gym.envs.gym_api

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models as models
import tools as tools
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd

dreamer_path = pathlib.Path(__file__).parent
util_path = dreamer_path.parent / "util"
sys.path.append(str(util_path))
from constants import Config


to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        
        # World Model (modified for vector observations)
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        
        # Task Behavior (continuous control focus)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        
        # Compilation (kept but not F1Tenth-specific)
        if config.compile and os.name != "nt":
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
            
        # Exploration (plan2explore recommended)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = {
            'greedy': lambda: self._task_behavior,
            'random': lambda: expl.Random(config, act_space),
            'plan2explore': lambda: expl.Plan2Explore(config, self._wm, reward)
        }[config.expl_behavior]().to(config.device)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            # Training logic (unchanged core)
            steps = self._config.pretrain if self._should_pretrain() else self._should_train(step)
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                # Removed video logging
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                self._logger.write(fps=True)

        # Policy execution (continuous actions only)
        policy_output, state = self._policy(obs, state, training)
        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        # Simplified for continuous control
        if state is None:
            latent = action = None
        else:
            latent, action = state
            
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)  # MLP encoder only
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
            
        feat = self._wm.dynamics.get_feat(latent)
        
        # Continuous action selection
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
            
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        
        # Removed discrete action conversion
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        # Core training remains unchanged
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            self._metrics.setdefault(name, []).append(value)


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def main(config):
    # Initialization (unchanged core)
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    # Directory setup
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    logger = tools.Logger(logdir, config.action_repeat * step)
    
    # Environment initialization
    print("Creating F1Tenth environments")
    train_envs = [gymnasium.make(id="SingleAgentAustria-v0", render_mode='rgb_array_follow') for _ in range(config.envs)]
    eval_envs = [gymnasium.make(id="SingleAgentAustria-v0", render_mode='rgb_array_follow') for _ in range(config.envs)]
    #! train and eval envs set to the same track for now
    
    # Parallel processing setup (unchanged)
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]

    # Action space setup (continuous specific)
    acts = train_envs[0].action_space
    print("Action Space", acts)
    config.num_actions = len(acts)

    # Dataset initialization (unchanged core)
    train_eps = tools.load_episodes(config.traindir, limit=config.dataset_size)
    eval_eps = tools.load_episodes(config.evaldir, limit=1)
    
    # Prefill with random actions (continuous specific)
    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps)")
        # Extract low/high for each action component in the Dict space
        action_lows = []
        action_highs = []
        for key in acts.spaces:
            action_lows.append(torch.tensor(acts[key].low))
            action_highs.append(torch.tensor(acts[key].high))

        # Concatenate lows/highs across action components
        action_low = torch.cat(action_lows).repeat(config.envs, 1)
        action_high = torch.cat(action_highs).repeat(config.envs, 1)

        random_actor = torchd.independent.Independent(
            torchd.uniform.Uniform(action_low, action_high), 1
        )

        # In the simulation lambda, return a dictionary with action keys
        state = tools.simulate(
            lambda o, d, s: ({
                "motor": random_actor.sample()[..., :1],       # Shape (envs, 1)
                "steering": random_actor.sample()[..., 1:]     # Shape (envs, 1)
            }, None),
            train_envs, train_eps, config.traindir, logger,
            limit=config.dataset_size, steps=prefill
        )

    # Agent setup (unchanged core)
    print("Initializing Dreamer agent")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    
    # Checkpoint loading (unchanged)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    # Training loop (modified for vector obs)
    while agent._step < config.steps + config.eval_every:
        # Evaluation phase
        if config.eval_episode_num > 0:
            print("Evaluating policy")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
            )
            
        # Training phase
        print("Training step")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
        )
        
        # Checkpoint saving
        torch.save({
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }, logdir / "latest.pt")

    # Cleanup
    for env in train_envs + eval_envs:
        try: env.close()
        except: pass
    return logdir

if __name__ == "__main__":
    config = Config()
    main(config)