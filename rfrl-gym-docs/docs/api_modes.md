# `rfrl_gym.modes`

[![Module](https://img.shields.io/badge/Module-rfrl__gym.modes-yellow?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_modes.md)

Reward-shaping wrappers — the **objective layer** of RFRL-Gym's four-part
modular stack. A mode wraps an already-composed environment to define the reward signal.

## `RewardMode` — base skeleton

[![Class](https://img.shields.io/badge/Class-RewardMode-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_modes.md#rewardmode)

`RewardMode` is not itself a usable reward function — its `reward()`
method raises `NotImplementedError`. It exists to define the contract
every reward mode must satisfy to stay compatible with the other three
layers:

- Wrap an already-composed environment via `Wrapper.__init__`, and leave
  the action, observation, termination, and info flow from `step()`
  untouched — only the reward is replaced.
- Compute reward from the action passed to `step()` and, when needed,
  ground-truth telemetry read from `env.unwrapped.info` — rather than
  private state kept on the wrapper itself.

Any custom reward mode should subclass `RewardMode` and implement
`reward()` following this contract, so it stays interchangeable with any
environment / detector / policy combination.

## `DSA` and `Jam` — reference implementations

[![Class](https://img.shields.io/badge/Class-DSA-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_modes.md#dsa)
[![Class](https://img.shields.io/badge/Class-Jam-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_modes.md#jam)

`DSA` and `Jam` ship with RFRL-Gym and double as templates for new reward
modes. Both follow the same shape:

1. Treat action `-1` as a no-op / backoff — reward `0`.
2. Otherwise, score the action against ground-truth state read from
   `env.unwrapped.info` at the current `step_number`.
3. Write the result, and a running cumulative total, back into the
   environment's own `reward_history` / `cumulative_reward` telemetry, so
   logging and rendering stay consistent regardless of which reward mode
   is active.

### `DSA`

[![Class](https://img.shields.io/badge/Class-DSA-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_modes.md#dsa)

Rewards **avoiding collisions** — models opportunistic dynamic
spectrum access:

| Action | Condition | Reward |
| ------ | --------- | ------ |
| `-1` | no-op / backoff | `0` |
| channel | unoccupied in `true_history` at current step | `+1` |
| channel | occupied in `true_history` at current step | `-1` |

### `Jam`

[![Class](https://img.shields.io/badge/Class-Jam-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_modes.md#jam)

Rewards **frequency alignment** — models an electronic-attack objective
against a specific target:

| Action | Condition | Reward |
| ------ | --------- | ------ |
| `-1` | no-op / backoff | `0` |
| channel | matches target entity's channel in `action_history` at current step | `+1` |
| channel | does not match | `-1` |

A custom objective (e.g. minimizing spectral footprint, or a continuous
SINR-based reward) follows the same pattern: subclass `RewardMode`, read
whatever ground-truth fields the objective needs from
`env.unwrapped.info`, and write back to `reward_history` /
`cumulative_reward` for consistency with the rest of the framework.

## Example

```python
from rfrl_gym.envs import RFRLGymIQEnv2
from rfrl_gym.modes import DSA

env = RFRLGymIQEnv2(scenario_filename="dsa_scenario.json", pywasp_config="pywasp.json")
env = DSA(env)  # add a detector wrapper here if the policy needs sensed observations

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

## See also

- [![Module](https://img.shields.io/badge/Module-rfrl__gym.envs-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_envs.md) — the environment layer that modes wrap.
- [![Module](https://img.shields.io/badge/Module-rfrl__gym.detectors-yellow?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_detectors.md) — the sensing layer, whose output a mode may read from.
- [RFRL-Gym overview](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_intro.md) — how all four layers fit together.


---

## API Reference

::: rfrl_gym.modes.reward_mode
    options:
        members:
          - RewardMode
          - DSA
          - Jam
        show_labels: true


