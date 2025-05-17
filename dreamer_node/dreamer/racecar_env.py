# Wrapper function allowing for racecar_gym env to be passed into Dreamer V3
import gymnasium
import numpy as np


class Racecar:
    metadata: dict = {}

    def __init__(self, train, visualize: bool = False):
        # TODO: Figure out how to make render_mode='human' if train env so we can actually see what is happening
        if train:
            self._env = gymnasium.make(
                id="SingleAgentRaceEnv-v0",
                scenario="gyms/racecar_gym/scenarios/austria.yml",
                render_mode="rgb_array_follow",  # optional
                render_options=dict(width=320, height=240, agent="A"),  # optional
            )
        else:
            self._env = gymnasium.make(
                id="SingleAgentRaceEnv-v0",
                scenario="gyms/racecar_gym/scenarios/austria.yml",
                render_mode="rgb_array_follow",  # optional
                render_options=dict(width=320, height=240, agent="A"),  # optional
            )

        if visualize:
            mode = "human"
        else:
            mode = "rgb_array_follow"
        self._env = gymnasium.make(
            id="SingleAgentRaceEnv-v0",
            scenario="gyms/racecar_gym/scenarios/austria.yml",
            render_mode=mode,
            render_options=dict(width=320, height=240, agent="A"),  # optional
        )

        self.reward_range = [-np.inf, np.inf]

        # Impose 100 step limit on environment until debugging is done
        self.steps_taken = 0
        self.step_limit = 100
        self.train = train

    @property
    def observation_space(self):
        base_obs_space = self._env.observation_space
        #! The Dreamer requires these additional keys to exist (see models.preprocess)
        # We do not actually use them but if we want to remove them, we would need to change Dreamer code
        spaces = {
            "image": gymnasium.spaces.Box(
                0, 255, shape=base_obs_space["hd_camera"].shape, dtype=np.uint8
            ),
            "is_first": gymnasium.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            "is_last": gymnasium.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            "is_terminal": gymnasium.spaces.Box(
                -np.inf, np.inf, (1,), dtype=np.float32
            ),
        }
        for k, v in base_obs_space.items():
            spaces[k] = v
        return gymnasium.spaces.Dict(spaces)

    @property
    def action_space(self):
        action_space = self._env.action_space
        action_space.discrete = False
        return action_space

    def step(self, action):
        if "action" in action:
            racecar_action = {}
            racecar_action["motor"] = action["action"][0]
            racecar_action["steering"] = action["action"][1]
        else:
            racecar_action = action
        base_obs, reward, done, truncated, info = self._env.step(racecar_action)

        self.steps_taken += 1
        if not self.train and self.steps_taken >= 100:
            truncated = True
            done = True

        obs = {
            "image": np.array(base_obs["hd_camera"], dtype=np.uint8),  # Ensure uint8
            "is_first": np.array([False], dtype=np.float32),
            "is_last": np.array([done], dtype=np.float32),
            "is_terminal": np.array([info.get("discount", 1.0) == 0], dtype=np.float32),
        }

        # Keep all original keys from the base environment
        for k, v in base_obs.items():
            if k != "hd_camera":  # Already mapped to "image"
                obs[k] = np.array(v, dtype=base_obs[k].dtype)

        return obs, np.float32(reward), done, truncated, info

    def render(self):
        return self._env.render()

    def reset(self, seed=None, options=None):
        base_obs, info = self._env.reset(seed=seed, options=options)
        self.steps_taken = 0

        obs = {
            "image": np.array(base_obs["hd_camera"], dtype=np.uint8),  # Ensure uint8
            "is_first": np.array([True], dtype=np.float32),
            "is_last": np.array([False], dtype=np.float32),
            "is_terminal": np.array([False], dtype=np.float32),
        }
        # Keep all original keys from the base environment
        for k, v in base_obs.items():
            if k != "hd_camera":  # Already mapped to "image"
                obs[k] = np.array(v, dtype=base_obs[k].dtype)
        return obs, info
