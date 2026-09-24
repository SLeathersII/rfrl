# RFRL-Gym

RFRL-Gym: a modular Gymnasium-based framework for RF reinforcement learning.

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

Each wrapper layer should be modular and interchanged functionally. A typical RFRL-Gym environment is
assembled by stacking wrappers around a base environment:

```python
env = gym.make('rfrl-gym-iq-v0.1', scenario_filename='sb3_test_scenario.json',
               pywasp_config = "pywaspgen/configs/default.json",
               num_episodes=1)     # 1. scene handler / IQ generator
env = OracleMap(env, 'detect')    # 2. sensing layer, shapes the observation
env = DSA(env)              # 3. reward mode, shapes the reward
# 4. policy/agent trains against `env` as a standard Gymnasium environment
```
Noting that wrappers are applied in the order they are wrapped and should be applied in the above order. A single IQ 
generation can then be functionally applied to process the scene (vectorized environment) with many different detectors 
or reward shaping compositions in parallel. 


See Also
--------
rfrl_gym.envs : Scene handlers / IQ generators.
rfrl_gym.detectors : Sensing wrappers that shape observations.
rfrl_gym.modes : Reward wrappers that shape the training signal.