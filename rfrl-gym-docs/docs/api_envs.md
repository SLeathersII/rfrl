# `rfrl_gym.envs`

[![Module](https://img.shields.io/badge/Module-rfrl__gym.envs-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_envs.md)

The environment layer owns the RF simulation itself: channelization,
entity bookkeeping, IQ data generation (where applicable), and the
Gymnasium action/observation space contract. Every class here is a
`gym.Env` subclass, constructed via `gym.make(...)` with a scenario file
and (for the IQ variants) a PyWaspGen config.

## Two paradigms in this module

The five environments split cleanly into two design generations, and
**which one you use changes how reward mode and detection get applied**:

| | Monolithic | Composable |
| --- | --- | --- |
| Classes | `RFRLGymAbstractEnv`, `RFRLGymAbstractEnv_LD`, `RFRLGymIQEnv` | `RFRLGymIQEnv2`, `RFRLGymIQEnv3` |
| Reward | Computed **inside** `step()` directly from `reward_mode` (`'dsa'`/`'jam'`) | Always returns `0` — real reward comes from a [`DSA`/`Jam`](./api_modes.md) wrapper applied outside the env |
| Sensing | Ground-truth `true_history` compared straight to the agent's action | Real observation space (raw IQ) — sensing comes from a [detector wrapper](./api_detectors.md) applied outside the env |
| Observation space | `Discrete` (encoded detect/classify integer) | `Box` (raw IQ samples, real + imaginary) |

In other words: the **monolithic environments are self-contained** — they
implement the full agent-facing contract themselves, including reward,
which is why the earlier example script (`sb3_example.py`) can call
`gym.make('rfrl-gym-abstract-v0', ...)` and train directly with no
wrappers. The **composable environments deliberately return a dummy
reward** so that `DSA(env)` / `Jam(env)` and a detector wrapper can be
layered on top — this is the [architecture described in the RFRL-Gym
overview](./api_intro.md), and `RFRLGymIQEnv2` is the environment that
`entity_generator.py`'s own docstring example uses.

## Shared setup pattern

All five environments load a scenario file (`scenario_filename`, read
from `scenarios/<file>.json`) and validate it (`num_channels > 0`,
`max_steps > 0`, `observation_mode` and `reward_mode` in the allowed
sets, `render_mode`/`render_fps` valid) before doing anything else. The
three IQ-based environments additionally load a `pywasp_config` JSON file
for `samples_per_step` and a random seed, and build their entity list via
[`EntityGenerator`](./api_entities.md).

---

## `RFRLGymAbstractEnv`

[![Class](https://img.shields.io/badge/Class-RFRLGymAbstractEnv-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_envs.md#rfrlgymabstractenv)
[![Paradigm](https://img.shields.io/badge/Paradigm-Monolithic-lightgrey?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_envs.md#rfrlgymabstractenv)

The original, self-contained abstract environment — no real IQ
generation, just ground-truth channel occupancy. Entities are
instantiated dynamically via a constructed-and-`eval`'d constructor
string (matching `type` against `rfrl_gym.entities`), and reward is
computed directly against `true_history`:

- **`reward_mode='dsa'`**: `+1` if the chosen channel is unoccupied, `-1` otherwise.
- **`reward_mode='jam'`**: `+1` if the chosen channel matches the target entity's channel, `-1` otherwise.

| Space | Shape |
| ----- | ----- |
| `action_space` | `Discrete(1 + num_channels)` — `0` reserved as a no-op/backoff action, matching the `-1` convention entities use internally. |
| `observation_space` | `Discrete(observation_base ** num_channels)` — `observation_base` is `2` for `'detect'` mode, `1 + num_entities` for `'classify'` mode. |

This is the environment used by [`sb3_example.py`](./api_getting_started.md#example-1--sb3_examplepy-abstract-gym-legacy-path) — `gym.make('rfrl-gym-abstract-v0', ...)`.



## `RFRLGymIQEnv`

[![Class](https://img.shields.io/badge/Class-RFRLGymIQEnv-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_envs.md#rfrlgymiqenv)
[![Paradigm](https://img.shields.io/badge/Paradigm-Monolithic-lightgrey?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_envs.md#rfrlgymiqenv)

The first real-IQ environment — introduces PyWaspGen-based signal
synthesis and an internal [`DetectorGenerator`](./api_detectors.md), but
still computes reward directly inside `step()` exactly like
`RFRLGymAbstractEnv` (same `dsa`/`jam` ±1 logic against `true_history`).
Builds one `BurstDef` per entity at construction, then re-randomizes each
burst's center frequency (within `modem_params['center_frequency']`)
every step while cycling which channel index is "current."

| Space | Shape |
| ----- | ----- |
| `action_space` | `Discrete(1 + num_channels)` |
| `observation_space` | `Discrete(observation_base ** num_channels)` — same detect/classify encoding as `RFRLGymAbstractEnv`. |

Effectively the transitional step between the abstract gym and the fully
composable IQ pipeline: real signal generation, but reward and sensing
are still baked into the environment rather than wrapped.

## `RFRLGymIQEnv2`

[![Class](https://img.shields.io/badge/Class-RFRLGymIQEnv2-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_envs.md#rfrlgymiqenv2)
[![Paradigm](https://img.shields.io/badge/Paradigm-Composable-blue?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_envs.md#rfrlgymiqenv2)

The current, wrapper-ready IQ environment — the one referenced
throughout the rest of this doc set (`api_intro.md`'s architecture
diagram, `scratch_train.py` in the getting-started guide). Two structural
changes from `RFRLGymIQEnv` make the wrapper pattern possible:

- **`step()` always returns reward `0`.** Real reward only exists once a
  [`DSA`](./api_modes.md#dsa) or [`Jam`](./api_modes.md#jam) wrapper is
  applied on top.
- **`observation_space` is a `Box`, not a `Discrete`.** Shape
  `(2, samples_per_step)` — raw real/imaginary IQ samples, not a
  detect/classify encoded integer. This is the raw signal a
  [detector wrapper](./api_detectors.md) (e.g. `CA_CFAR`) processes into
  an actual sensed observation.

Ground truth is computed per-step via `_get_true_step_occupancy()`
(reading straight from `action_history`, independent of reward or
sensing), and bursts are regenerated fresh each step through `iq_gen()`
rather than mutated in place.

| Space | Shape |
| ----- | ----- |
|

## `RFRLGymAbstractEnv_LD`

[![Class](https://img.shields.io/badge/Class-RFRLGymAbstractEnv__LD-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_envs.md#rfrlgymabstractenv_ld)
[![Paradigm](https://img.shields.io/badge/Paradigm-Monolithic-lightgrey?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_envs.md#rfrlgymabstractenv_ld)
[![Research](https://img.shields.io/badge/Research-Learning%20Dynamics-orange?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_envs.md#rfrlgymabstractenv_ld)

A stripped-down variant built for studying **learning dynamics** rather
than realistic scenarios — its own docstring describes it as an "upper
bound" baseline. Instead of loading a scenario file and simulating
entities, it pre-generates a fixed pool of `num_states` unique
channel-occupancy patterns (`gen_states`, combinatorially sampled with a
seed) and cycles through them by rolling the state array each step —
there's no entity behavior at all, just a perfectly-sensed, repeating
grid-world label map.

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `num_channels` | `int` | `10` | Channel count. |
| `num_emitters` | `int` | `7` | How many channels are "occupied" in each generated state. |
| `num_states` | `int` | `10` | Size of the pre-generated state pool (bounded by `C(num_channels, num_emitters)`). |
| `max_steps` | `int` | `500` | Episode length. |
| `render_mode` | `str` | `'null'` | Same options as other environments. |
| `reward_mode` | `str` | `'dsa'` | `'dsa'` or `'jam'`. |
| `seed` | `int` | `3` | Seed for state-pool generation. |

> **Caveat:** this class's own comments flag unresolved rough edges — the
> `target_entity` is hardcoded to `"fixed_hop_freq"` with a `# TODO try
> and remove/avoid needing at this level` note, and `'jam'` reward mode
> references `self.target_idx`, which is never actually assigned in this
> class (only `'dsa'` mode is confirmed functional here). Treat this
> environment as an experimental research tool rather than a
> general-purpose scenario runner.
::: rfrl_gym.envs
    options:
        members: true
        show_labels: true
