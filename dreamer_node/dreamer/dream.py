import argparse
import os
import sys
import pathlib
import functools
from typing import Generator, NoReturn, Any, Dict, Tuple, List

import numpy as np
import gymnasium.spaces

import torch
from torch import nn
from torch import distributions as torchd

import tools  # type: ignore
import models  # type: ignore
from config import Config  # type: ignore
import exploration as expl  # type: ignore
from racecar_env import Racecar  # type: ignore
from parallel import Parallel, Damy  # type: ignore

sys.path.append(str(pathlib.Path(__file__).parent.parent))
import racecar_gym.envs.gym_api  # type: ignore # noqa: F401


def to_np(x):
    return x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(
        self,
        obs_space: gymnasium.spaces.Dict,
        act_space: gymnasium.spaces.Dict,
        config: Config,
        logger: tools.Logger,
        dataset: Generator[dict, Any, NoReturn],
    ):
        super(Dreamer, self).__init__()
        self._config = config
        print(f'config initialized to {config}')
        self._logger = logger
        print(f'logger initialized to {logger}')
        self._should_log = tools.Every(config.LOG_EVERY)
        batch_steps = config.BATCH_SIZE * config.BATCH_LENGTH
        self._should_train = tools.Every(batch_steps / config.TRAIN_RATIO)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.RESET_EVERY)
        self._should_expl = tools.Until(
            int(config.EXPLORATION_UNTIL / config.ACTION_REPEAT)
        )
        self._metrics: Dict[str, int | list] = {}
        self._step = logger.step // config.ACTION_REPEAT
        self._update_count = 0
        print(f'update count initialized to {_update_count}')
        self._dataset = dataset
        print(f'data set initialized to {_dataset}')

        # World Model (modified for vector observations)
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)

        # Task Behavior (continuous control focus)
        self._task_behavior = models.ImagBehavior(config, self._wm)

        # Compilation (kept but not F1Tenth-specific)
        if config.COMPILE and os.name != "nt":
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)

        # Exploration (plan2explore recommended)
        def reward(f, s, a):
            return self._wm.heads["reward"](f).mean()

        self._expl_behavior: nn.Module = {
            "greedy": lambda: self._task_behavior,
            "random": lambda: expl.Random(config, act_space),
            "plan2explore": lambda: expl.Plan2Explore(config, self._wm, reward),
        }[config.EXPLORATION_BEHAVIOR]().to(config.DEVICE)

    def __call__(
        self, obs, reset, state=None, training: bool = True
    ) -> Tuple[dict, Tuple[dict, Any]]:
        step = self._step

        if training:
            print('training...')
            # Training logic (unchanged core)
            steps = (
                self._config.PRETRAIN
                if self._should_pretrain()
                else self._should_train(step)
            )

            for _ in range(steps):
                # print("SAMPLED DATASET")
                # next_dataset = next(self._dataset)
                # print(next_dataset)
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
            self._logger.step = self._config.ACTION_REPEAT * self._step

        return policy_output, state

    def _policy(self, obs, state, training):
        # Simplified for continuous control
        if state is None:
            latent = action = None
            print(f'latent & action= None')
        else:
            latent, action = state
            print(f'latent & action& state= {state}')

        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)  # MLP encoder only
        # print("----------")
        # print("latent")
        # print(latent)
        # print("----------")
        # print("action")
        # print(action)
        # print("----------")
        # print("embed")
        # print(embed.shape)
        # print("----------")
        # print("is_first")
        # print(obs["is_first"])
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])

        if self._config.EVALUATION_STATE_MEAN:
            latent["stoch"] = latent["mean"]

        feat = self._wm.dynamics.get_feat(latent)

        # Continuous action selection
        if not training:
            actor = self._task_behavior.actor(feat)
            print(f'actor: {actor}')
            action = actor.mode()
            print(f'action{action}')

        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            print(f'actor: {actor}')
            action = actor.sample()
            print(f'action: {action}')

        else:
            actor = self._task_behavior.actor(feat)
            print(f'actor: {actor}')
            action = actor.sample()
            print(f'action{action}')

        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        print(f'latent: {latent}')
        action = action.detach()
        print(f'action: {action}')

        # Removed discrete action conversion
        policy_output = {"action": action, "logprob": logprob}
        print(f'policy output: {policy_output}')
        state = (latent, action)
        print(f'state: {state}')
        return policy_output, state

    def _train(self, data):
        # Core training remains unchanged
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        print(f'metrics: {metrics}')
        start = post
        print(f'start: {start}')

        def reward(f, s, a):
            return self._wm.heads["reward"](self._wm.dynamics.get_feat(s)).mode()

        metrics.update(self._task_behavior._train(start, reward)[-1])
        print(f'metrics: {metrics}')

        if self._config.EXPLORATION_BEHAVIOR != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
            print(f'metrics: {metrics}')

        for name, value in metrics.items():
            self._metrics.setdefault(name, []).append(value)
        print(f'metrics: {metrics}')


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config: Config):
    generator = tools.sample_episodes(episodes, config.BATCH_LENGTH)
    dataset = tools.from_generator(generator, config.BATCH_SIZE)
    print(f'dataset: {dataset}')
    return dataset


def main(config: Config):
    # Initializing log directories (unchanged)
    tools.set_seed_everywhere(config.SEED)
    if config.DETERMINISTIC_RUN:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.LOG_DIRECTORY).expanduser()
    config.TRAINING_DIRECTORY = config.TRAINING_DIRECTORY or logdir / "train_eps"
    config.EVALUATION_DIRECTORY = config.EVALUATION_DIRECTORY or logdir / "eval_eps"
    config.STEPS //= config.ACTION_REPEAT
    config.EVALUATION_EVERY //= config.ACTION_REPEAT
    config.LOG_EVERY //= config.ACTION_REPEAT
    config.TIME_LIMIT //= config.ACTION_REPEAT

    # Directory setup (unchanged)
    logdir.mkdir(parents=True, exist_ok=True)
    config.TRAINING_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config.EVALUATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.TRAINING_DIRECTORY)
    print(f'step: {step}')
    logger = tools.Logger(logdir, config.ACTION_REPEAT * step)
    print(f'logger: {logger}')

    # GUI initialization
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI")
    args = parser.parse_args()

    # Environment initialization
    print("Creating F1Tenth environments")
    train_envs: List[Racecar] | List[Parallel] | List[Damy] = [
        Racecar(train=True, visualize=args.gui) for _ in range(config.ENVIRONMENT_COUNT)
    ]
    eval_envs: List[Racecar] | List[Parallel] | List[Damy] = [
        Racecar(train=False, visualize=args.gui)
        for _ in range(config.ENVIRONMENT_COUNT)
    ]
    #! train and eval envs set to the same track for now, may want to change later

    # Parallel processing setup (unchanged)
    if config.PARALLEL:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]

    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]

    # Action space setup (continuous)
    acts = train_envs[0].action_space
    print("Action Space", acts)
    config.num_actions = len(acts)

    # Dataset initialization (unchanged)
    train_eps = tools.load_episodes(
        config.TRAINING_DIRECTORY, limit=config.DATASET_SIZE
    )
    eval_eps = tools.load_episodes(config.EVALUATION_DIRECTORY, limit=1)

    # Prefill with random actions (continuous)
    state = None
    if not config.OFFLINE_TRAINING_DIRECTORY:
        prefill = max(0, config.PREFILL - count_steps(config.TRAINING_DIRECTORY))
        print(f"Prefill dataset ({prefill} steps)")

        # Extract low/high for each action component in the Dict space
        action_lows: List[torch.Tensor] = []
        action_highs: List[torch.Tensor] = []
        for key in acts.spaces:
            action_lows.append(torch.tensor(acts[key].low))
            action_highs.append(torch.tensor(acts[key].high))
        print(f'action lows: {action_lows}')
        print(f'action highs: {action_highs}')
        # Concatenate lows/highs across action components
        action_low = torch.cat(action_lows).repeat(config.ENVIRONMENT_COUNT, 1)
        action_high = torch.cat(action_highs).repeat(config.ENVIRONMENT_COUNT, 1)
        random_actor = torchd.independent.Independent(
            torchd.uniform.Uniform(action_low, action_high), 1
        )
        print(f'action lows: {action_lows}')
        print(f'action highs: {action_highs}')

        # In the simulation lambda, return a dictionary with action keys
        state = tools.simulate(
            lambda o, d, s: (
                {
                    "motor": random_actor.sample()[..., :1],  # Shape (envs, 1)
                    "steering": random_actor.sample()[..., 1:],  # Shape (envs, 1)
                },
                None,
            ),
            train_envs,
            train_eps,
            config.TRAINING_DIRECTORY,
            logger,
            limit=config.DATASET_SIZE,
            steps=prefill,
        )

    # Agent setup (unchanged)
    print("Initializing Dreamer agent")
    train_dataset = make_dataset(train_eps, config)
    # eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.DEVICE)

    # Checkpoint loading (unchanged)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    # Training loop (modified for vector obs)
    while agent._step < config.STEPS + config.EVALUATION_EVERY:
        # Logging
        progress_percent = (agent._step / config.STEPS) * 100
        print(f"Training progress: {progress_percent:.2f}%")
        logger.scalar("training_progress", progress_percent)
        logger.write(step=agent._step)

        # Evaluation phase
        if config.EVALUATION_EPISODE_NUMBER > 0:
            print("Evaluating policy")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.EVALUATION_DIRECTORY,
                logger,
                is_eval=True,
                episodes=config.EVALUATION_EPISODE_NUMBER,
            )

        # Training phase
        print("Training step")

        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.TRAINING_DIRECTORY,
            logger,
            limit=config.DATASET_SIZE,
            steps=config.EVALUATION_EVERY,
            state=state,
        )

        # Checkpoint saving
        torch.save(
            {
                "agent_state_dict": agent.state_dict(),
                "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            },
            logdir / "latest.pt",
        )

    # Cleanup
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception as e:
            print(e)
    return logdir


if __name__ == "__main__":
    config = Config()
    main(config)
