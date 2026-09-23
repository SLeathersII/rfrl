"""Reward-shaping wrappers for the RFRL Gym.

Re-exports the reward mode classes from :mod:`rfrl_gym.modes.reward_mode`
so they're available directly as ``rfrl_gym.modes.RewardMode``,
``rfrl_gym.modes.DSA``, and ``rfrl_gym.modes.Jam``.
"""

from .reward_mode import RewardMode, DSA, Jam

#__all__ = ["RewardMode", "DSA", "Jam"]
