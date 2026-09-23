# Reward-shaping wrappers for the RFRL Gym.

This module defines :class:`RewardMode`, a Gymnasium ``Wrapper`` base class,
and two concrete objective functions built on top of it (:class:`DSA` and
:class:`Jam`). None of these classes are standalone environments: each one
wraps an *underlying* RF simulator/environment instance (e.g.
``RFRLGymIQEnv2``) and intercepts its :meth:`~gymnasium.Env.step` output to
replace the scalar reward with a task-specific objective. The wrapped
environment is still the one running the RF simulation, tracking channel
occupancy, and advancing ``step_number``; these wrappers only reshape the
learning signal the agent receives on top of it.
"""

# modes API

::: rfrl_gym.modes.reward_mode
    options:
        members:
          - RewardMode
          - DSA
          - Jam
        show_labels: true


