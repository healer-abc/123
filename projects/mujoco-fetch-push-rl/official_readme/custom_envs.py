"""Project extension loaded by the unchanged RL Zoo import_envs module."""
import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)


class ActionGain(gym.Wrapper):
    """One independently sampled gain per episode; observation/reward unchanged."""

    def __init__(self, env, gain_low=1.0, gain_high=1.0):
        super().__init__(env)
        if not 0 < gain_low <= gain_high:
            raise ValueError("Expected 0 < gain_low <= gain_high")
        self.gain_low, self.gain_high = gain_low, gain_high
        self.gain_rng = np.random.default_rng()
        self.gain = 1.0

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self.gain_rng = np.random.default_rng(np.random.SeedSequence([seed, 1729]))
        self.gain = float(self.gain_rng.uniform(self.gain_low, self.gain_high))
        return obs, info

    def step(self, action):
        executed = np.asarray(action, dtype=np.float32).copy()
        executed[:3] = np.clip(executed[:3] * self.gain, -1, 1)
        obs, reward, terminated, truncated, info = self.env.step(executed)
        info["action_gain"] = self.gain
        return obs, float(reward), terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.unwrapped.compute_reward(achieved_goal, desired_goal, info)


def make_fetch_push_gain(gain_low=0.8, gain_high=1.2, **kwargs):
    return ActionGain(gym.make("FetchPush-v4", **kwargs), gain_low, gain_high)


if "FetchPushGain-v0" not in gym.registry:
    gym.register(
        id="FetchPushGain-v0",
        entry_point="custom_envs:make_fetch_push_gain",
        max_episode_steps=50,
    )
