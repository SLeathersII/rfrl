# Reward-shaping wrappers for the RFRL Gym.

RFRL-Gym reward-shaping wrappers: the objective-function layer of a four-part modular stack.

### `RewardMode` as a base skeleton

:class:`RewardMode` is not itself a usable reward function --
:meth:`RewardMode.reward` raises :class:`NotImplementedError`. It defines
the contract every reward mode must satisfy to stay compatible with the
other three components:

- Wrap an already-composed environment (env, optionally with a detector
  already applied) via ``Wrapper.__init__``, and leave the action,
  observation, termination and info flow from :meth:`~gymnasium.Env.step`
  untouched -- only the reward is replaced.
- Compute reward from the action passed to ``step`` and, when needed,
  ground-truth telemetry read from ``env.unwrapped.info``, rather than
  private state kept on the wrapper itself.

Any custom reward mode should subclass :class:`RewardMode` and implement
:meth:`RewardMode.reward` following this contract, so it remains
interchangeable with any environment/detector/policy combination.

### `DSA` and `Jam` as reference implementations

:class:`DSA` and :class:`Jam` ship with RFRL-Gym and double as templates
for new reward modes. Both follow the same shape:

1. Treat action ``-1`` as a no-op/backoff, reward ``0``.
2. Otherwise, score the action against ground-truth state read from
   ``env.unwrapped.info`` at the current ``step_number``.
3. Write the result, and a running cumulative total, back into the
   environment's own ``reward_history`` / ``cumulative_reward`` telemetry,
   so logging and rendering stay consistent regardless of which reward
   mode is active.

:class:`DSA` rewards *avoiding collisions*: +1 for transmitting on a
channel unoccupied in ``true_history`` at the current step, -1 for an
occupied one -- modeling opportunistic, non-cooperative spectrum access.
:class:`Jam` rewards *frequency alignment*: +1 when the agent's action
matches the target entity's channel in ``action_history`` at the current
step, -1 otherwise -- modeling an electronic-attack objective against a
specific target. A custom objective (e.g. minimizing spectral footprint,
or a continuous SINR-based reward) follows the same pattern: subclass
:class:`RewardMode`, read whatever ground-truth fields the objective
needs from ``env.unwrapped.info``, and write back to ``reward_history`` /
``cumulative_reward`` for consistency with the rest of the framework.

Examples
--------
>>> from rfrl_gym.envs import RFRLGymIQEnv2
>>> from rfrl_gym.modes import DSA
>>> env = RFRLGymIQEnv2(scenario_filename="dsa_scenario.json", pywasp_config="pywasp.json")
>>> env = DSA(env)  # add a detector wrapper here if the policy needs sensed observations
>>> obs, info = env.reset()
>>> obs, reward, terminated, truncated, info = env.step(action)

See Also
--------
RewardMode : Abstract base class defining the reward-mode contract.
DSA : Reference reward mode for opportunistic spectrum access.
Jam : Reference reward mode for target-frequency jamming.

# modes API

::: rfrl_gym.modes.reward_mode
    options:
        members:
          - RewardMode
          - DSA
          - Jam
        show_labels: true


