# `rfrl_gym.entities`

[![Module](https://img.shields.io/badge/Module-rfrl__gym.entities-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_entities.md)

Entities are the RF actors that populate an RFRL-Gym scenario — every
transmitter, hopper, or jammer simulated in the environment. Each one is an independent, exogenous state machine: on every
step it decides which channel (if any) to occupy, and the environment
composes all active entities' emissions into the scene's baseband IQ
data via PyWaspGen.

Entities are declared entirely through the [`entities` block of a
scenario file](./scenario_configs.md#entities) — `EntityGenerator` reads
each entry's `type`, matches it against the registry below, and
constructs it with the parameters you provide.

## `Entity` — shared base contract

[![Class](https://img.shields.io/badge/Class-Entity-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_entities.md#entity)

Every entity subclasses `Entity`, which owns three responsibilities
common to all of them:

- **Construction and validation** — accepts `entity_label`, `num_channels`,
  `channels`, `onoff`, `start`, `stop`, and `modem_params`, then validates
  structural constraints (e.g. every value in `channels` must fall within
  `[0, num_channels)`).
- **The duty-cycle state machine** — `onoff = [x, y, z]` drives a simple
  on/off timer: start in state `x`, stay active for `y` steps, stay
  dormant for `z` steps, repeat. This runs identically for every entity
  regardless of subclass logic.
- **The step/reset interface** — `get_action(info)` is called once per
  environment step and returns either a channel index or `-1` (no
  transmission); `reset(info)` re-initializes state at episode boundaries.
  Both delegate to subclass-specific hooks (`_get_action()`, `_reset()`,
  `_validate_self()`) that every concrete entity must implement.

`Entity` itself is abstract — instantiating it directly and calling
`_get_action()` raises `NotImplementedError`.

## Registry

Six concrete entity types are currently registered, split into two
behavioral categories used internally by `EntityGenerator`:

| Category | Classes | Why |
| -------- | ------- | --- |
| **Deterministic** | `ConstantFreq`, `FixedHopFreq` | Trajectory depends only on time/internal counters — fully pre-computable. |
| **Dynamic** | `StochasticConstantFreq`, `StochasticHopFreq`, `AgileFreq`, `SimpleJammer` | Reactive or stochastic — must be computed step-by-step against live environment state. |

### `ConstantFreq`

[![Class](https://img.shields.io/badge/Class-ConstantFreq-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_entities.md#constantfreq)
[![Deterministic](https://img.shields.io/badge/Deterministic-8884-lightgrey?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_entities.md#constantfreq)

The always-on, non-stochastic baseline emitter on a fixed channel — no
parameters beyond the base `Entity` contract.

### `FixedHopFreq`

[![Class](https://img.shields.io/badge/Class-FixedHopFreq-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_entities.md#fixedhopfreq)
[![Deterministic](https://img.shields.io/badge/Deterministic-8884-lightgrey?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_entities.md#fixedhopfreq)

Models a traditional FHSS transceiver — cycles through `channels` in
order, or through a fixed shuffled permutation chosen once at
construction if `rand_hop=1` (default).

| Parameter | Type | Default |
| --------- | ---- | ------- |
| `rand_hop` | `int` (`{0,1}`) | `1` |

### `StochasticConstantFreq`

[![Class](https://img.shields.io/badge/Class-StochasticConstantFreq-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_entities.md#stochasticconstantfreq)
[![Dynamic](https://img.shields.io/badge/Dynamic-8884-orange?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_entities.md#stochasticconstantfreq)

Models an intermittent primary user or bursty node on a single fixed
channel (`channels[0]`). Each active step is independently gated by a
Bernoulli draw against `percent_on`, overlaid on the base duty cycle
rather than replacing it.

| Parameter | Type | Default |
| --------- | ---- | ------- |
| `percent_on` | `float`, `[0,1]` | `1.0` |

### `StochasticHopFreq`

[![Class](https://img.shields.io/badge/Class-StochasticHopFreq-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_entities.md#stochastichopfreq)
[![Dynamic](https://img.shields.io/badge/Dynamic-8884-orange?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_entities.md#stochastichopfreq)

The stochastic counterpart to `FixedHopFreq`. Currently flagged by the
registry itself as non-seed-reproducible (`TODO — w/seed=deterministic`).

### `AgileFreq`

[![Class](https://img.shields.io/badge/Class-AgileFreq-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_entities.md#agilefreq)
[![Dynamic](https://img.shields.io/badge/Dynamic-8884-orange?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_entities.md#agilefreq)

An opportunistic-spectrum-access node: reads the prior step's
ground-truth occupancy and the agent's prior action, and hops to a
randomly chosen vacant channel if its current one was collided with.
Takes no parameters beyond the base contract. **Note:** its `channels`
restriction is not currently enforced by the hop logic.

### `SimpleJammer`

[![Class](https://img.shields.io/badge/Class-SimpleJammer-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_entities.md#simplejammer)
[![Dynamic](https://img.shields.io/badge/Dynamic-8884-orange?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_entities.md#simplejammer)

A naive spot-jammer with no targeting or prediction logic — samples
uniformly each step from whichever channels were occupied in the prior
step's ground truth.

| Parameter | Type | Default |
| --------- | ---- | ------- |
| `avoid_repeats` | `bool` | `True` |

## See also

- [Scenario configuration files](./scenario_configs.md) — the `entities` block schema that instantiates these classes.
- [![Module](https://img.shields.io/badge/Module-rfrl__gym.envs-green?style=for-the-badge)](./api_envs.md) — consumes the entity list to synthesize scene IQ data.
- [RFRL-Gym overview](./api_intro.md) — how entities fit into the full four-layer stack.

---

## API Reference

::: rfrl_gym.entities
    options:
        members: true
        show_labels: true

