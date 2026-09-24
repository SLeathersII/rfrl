"""RFRL-Gym: a modular Gymnasium-based framework for RF reinforcement learning.

RFRL-Gym decomposes an RF reinforcement-learning problem into four
independently swappable components, each implemented as a layer that wraps
the one below it:

1. :mod:`rfrl_gym.envs` -- the **environment / scene handler**. Owns the RF
   simulation itself: IQ data generation, channel state, entity behavior,
   and scenario bookkeeping (e.g. :class:`~rfrl_gym.envs.RFRLGymIQEnv2`).
2. :mod:`rfrl_gym.detectors` -- the **sensing layer**. A wrapper that turns
   the environment's raw IQ output into the observation an agent actually
   sees (e.g. detection or classification results).
3. :mod:`rfrl_gym.modes` -- the **objective layer**. A wrapper that
   replaces the environment's scalar reward with a task-specific objective
   (e.g. :class:`~rfrl_gym.modes.DSA`, :class:`~rfrl_gym.modes.Jam`),
   computed from environment ground truth rather than from what the
   detector exposes.
4. **Policy / agent** -- the learning algorithm, external to this package,
   that consumes the detector's observations and the reward mode's reward
   to select actions.

Each layer depends only on the *interface* of the layer beneath it, never
its internal implementation, which is what makes environments, detectors,
and reward modes freely recombinable. A typical RFRL-Gym environment is
assembled by stacking wrappers around a base environment:

```python
env = RFRLGymIQEnv2(scenario_filename="scenario.json", pywasp_config="pywasp.json")
env = SomeDetector(env)   # sensing layer: shapes the observation
env = DSA(env)            # objective layer: shapes the reward
# a policy/agent then trains against `env` as a standard Gymnasium environment
```

Because reward is never intrinsic to an environment, every base RFRL-Gym
environment must be paired with a reward-mode wrapper from
:mod:`rfrl_gym.modes` before it produces a usable training signal.

See Also
--------
rfrl_gym.envs : Scene handlers / IQ generators.
rfrl_gym.detectors : Sensing wrappers that shape observations.
rfrl_gym.modes : Reward wrappers that shape the training signal.
"""

import numpy as np
from gymnasium.envs.registration import register
from . import modes
from . import entities
from . import detectors


register(
    id='rfrl-gym-abstract-v0',
    entry_point='rfrl_gym.envs:RFRLGymAbstractEnv',
    max_episode_steps = 1000
)

register(
    id='rfrl-gym-abstract-v0.1',
    entry_point='rfrl_gym.envs:RFRLGymAbstractEnv_LD',
    max_episode_steps = 1000
)

register(
    id='rfrl-gym-iq-v0',
    entry_point='rfrl_gym.envs:RFRLGymIQEnv',
    max_episode_steps = 1000
)

register(id='rfrl-gym-iq-v0.1',
    entry_point='rfrl_gym.envs:RFRLGymIQEnv2',
    max_episode_steps = 1000)

register(id='rfrl-gym-wild-iq-v0',
    entry_point='rfrl_gym.envs:RFRLGymIQEnv3',
    max_episode_steps = 1000)