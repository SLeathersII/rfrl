# Scenario Configuration Files

[![Guide](https://img.shields.io/badge/Docs-Scenario%20Configs-blue?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/scenario_configs.md)

A **scenario file** is a JSON document that fully specifies an RFRL-Gym
episode: how many channels exist, which entities occupy the spectrum, what
the agent is trying to accomplish, and how the environment should render.
It's passed to the environment constructor via `scenario_filename`:

```python
env = gym.make(
    'rfrl-gym-iq-v0.1',
    scenario_filename='dsa_scenario.json',
    pywasp_config='pywaspgen/configs/default.json',
    num_episodes=1,
)
```

Every scenario file must contain exactly three top-level keys:
`environment`, `entities`, and `render`.

```json
{
  "environment": { "...": "..." },
  "entities":    { "...": "..." },
  "render":      { "...": "..." }
}
```

---

## `environment`

Global parameters that define the RF simulation and the RL objective for
the episode.

| Key | Type | Required | Description |
| --- | ---- | -------- | ----------- |
| `num_channels` | `int` | ✅ | Number of RF channels available in the environment. |
| `max_steps` | `int` | ✅ | Maximum number of steps allowed per episode. |
| `observation_mode` | `str` | ✅ | Sensing mode — must be `"detect"` or `"classify"`. |
| `reward_mode` | `str` | ✅ | Objective for the episode — must be `"dsa"` or `"jam"`. Maps to the [`DSA`](./api_modes.md#dsa) / [`Jam`](./api_modes.md#jam) reward-mode classes. |
| `target_entity` | `str` | ✅ | String identifier of the target entity. Must exactly match one of the keys defined in the `entities` block below. |

```json
"environment": {
  "num_channels": 8,
  "max_steps": 200,
  "observation_mode": "detect",
  "reward_mode": "dsa",
  "target_entity": "incumbent_1"
}
```

---

## `entities`

A dictionary describing every non-agent entity present in the scenario.
**Each key is an entity's label** (a string you choose, referenced
elsewhere in the file — e.g. by `environment.target_entity`), and each
value is that entity's own parameter dictionary.

| Key | Type | Required | Description |
| --- | ---- | -------- | ----------- |
| `type` | `str` | ✅ | The exact class name as it appears in `rfrl_gym.entities` (see the registry below). |
| *(anything else)* | *any* | — | Every other key is passed straight through as a keyword argument to that entity's constructor. |

Under the hood, `EntityGenerator` parses this block directly: for each
entity it automatically injects `entity_label` (the JSON key) and
`num_channels` (copied from `environment.num_channels`) into the
constructor call, then forwards every other key in the entity's dict as an
additional keyword argument. If `type` doesn't match a registered class,
construction fails immediately with `NotImplementedError` naming every
valid type.

```json
{
    "environment": 
    {
	    "num_channels": 10,
	    "max_steps": 100,
	    "observation_mode": "detect",
	    "reward_mode": "jam",
        "target_entity": "fixed_hop_freq_1",
        "detector": 
        {
            "energy_detector":
            {
                "type": "EnergyDetector"
            }
        }    
    },
    "entities": 
    {
        "constant_freq_1": 
        {
            "type": "ConstantFreq",
            "channels": [0],
            "onoff": [1,1,0],
            "modem_params":
            {
                "type": "qam",
                "order": 16,
                "filter": "RRC",
                "center_frequency": [-0.1,0.1],
                "bandwidth": 0.25,
                "start": 0.25,
                "duration": 0.25
            }
        },
        "constant_freq_2":
        {
            "type": "ConstantFreq",
            "channels": [9],
            "onoff": [1,1,0],
            "modem_params":
            {
                "type": "psk",
                "order": 4,
                "filter": "RRC",
                "center_frequency": [-0.1,0.1],
                "bandwidth": 0.5,
                "start": 0.5,
                "duration": 0.25
            }
        },
        "fixed_hop_freq_1":
        {
            "type": "FixedHopFreq",
            "channels": [1,2,3,4,5,6,7,8],
            "onoff": [1,1,0],
            "rand_hop": 0,
            "modem_params":
            {
                "type": "n_fmcw",
                "center_frequency": [0.0,0.0],
                "bandwidth": 1.0,
                "start": 0.0,
                "duration": 1.0
            }
        },
        "fixed_hop_freq_2":
        {
            "type": "FixedHopFreq",
            "channels": [3,6,8,1,4,2,5,7],
            "onoff": [1,1,0],
            "rand_hop": 0,
            "modem_params":
            {
                "type": "ask",
                "order": 4,
                "filter": "RRC",
                "center_frequency": [0.0,0.0],
                "bandwidth": 0.5,
                "start": 0.0,
                "duration": 1.0
            }
        }          
    },
    "render":
    {
        "render_mode": "pyqt",
        "render_fps": 20,
        "render_history": 20,
        "render_background": "black"
    }
}
```

### The `Entity` base contract

Every entity class shares this constructor signature and duty-cycle
mechanism:

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `entity_label` | `str` | — | Unique label for this instance (injected automatically from the JSON key). |
| `num_channels` | `int` | — | Total channel count (injected automatically from `environment.num_channels`). |
| `channels` | `list[int]` | — | Subset of channel indices this entity may use. Must fall within `[0, num_channels)`. |
| `onoff` | `list[int]` (`[x, y, z]`) | `[1, 1, 0]` | Duty-cycle state machine: `x` = initial state (0 dormant / 1 active), `y` = active-window length in steps, `z` = dormant-window length in steps. |
| `start` | `int` or `None` | `None` → `0` | Step at which the entity begins executing. |
| `stop` | `int` or `None` | `None` → `∞` | Step at which the entity permanently stops. |
| `modem_params` | `dict` | — | PyWaspGen physical-layer config: `type` (modulation, e.g. `"qam"`, `"psk"`, `"fsk"`), `order` (constellation size), `filter` (e.g. `"RRC"`), `center_frequency` (`[low, high]` in `[-0.5, 0.5]`), `bandwidth`, `start`, `duration` (fractional, in `(0, 1)`). |

At each step, an entity outside its `[start, stop]` window always returns
`-1` (no-op), regardless of subclass logic.

### Entity classes

RFRL-Gym registers six entity types, split into two behavioral categories:
**deterministic** entities (trajectory can be fully pre-computed) and
**dynamic** entities (reactive/stochastic, computed step-by-step).

#### `ConstantFreq`

[![Class](https://img.shields.io/badge/Class-ConstantFreq-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_entities.md#constantfreq)
[![Deterministic](https://img.shields.io/badge/Deterministic-8884-lightgrey?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_entities.md#constantfreq)

The always-on, non-stochastic baseline — a fixed-frequency emitter with no
parameters beyond the base `Entity` contract.

#### `FixedHopFreq`

[![Class](https://img.shields.io/badge/Class-FixedHopFreq-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_entities.md#fixedhopfreq)
[![Deterministic](https://img.shields.io/badge/Deterministic-8884-lightgrey?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_entities.md#fixedhopfreq)

Models a traditional FHSS transceiver — cycles through `channels` in
order, or through a fixed shuffled permutation of them (chosen once, at
construction) if `rand_hop=1`.

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `rand_hop` | `int` (`{0, 1}`) | `1` | `1` = shuffle `channels` once at init, then hop through that fixed order; `0` = hop in the order given. |

#### `StochasticConstantFreq`

[![Class](https://img.shields.io/badge/Class-StochasticConstantFreq-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_entities.md#stochasticconstantfreq)
[![Dynamic](https://img.shields.io/badge/Dynamic-8884-orange?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_entities.md#stochasticconstantfreq)

Models an intermittent primary user or bursty node on a single fixed
channel (`channels[0]`). Each active step is independently gated by a
Bernoulli draw against `percent_on`, layered on top of the base duty
cycle rather than replacing it.

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `percent_on` | `float`, `[0, 1]` | `1.0` | Per-step transmission probability. `1.0` behaves identically to `ConstantFreq` whenever the base duty cycle is active. |

#### `StochasticHopFreq`

[![Class](https://img.shields.io/badge/Class-StochasticHopFreq-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_entities.md#stochastichopfreq)
[![Dynamic](https://img.shields.io/badge/Dynamic-8884-orange?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_entities.md#stochastichopfreq)

The stochastic counterpart to `FixedHopFreq`. Currently listed by the
framework itself as non-seed-reproducible (`TODO — w/seed=deterministic`
in the entity registry).

#### `AgileFreq`

[![Class](https://img.shields.io/badge/Class-AgileFreq-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_entities.md#agilefreq)
[![Dynamic](https://img.shields.io/badge/Dynamic-8884-orange?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_entities.md#agilefreq)

An opportunistic-spectrum-access node: reads the prior step's
ground-truth occupancy and the agent's prior action, and if its current
channel was collided with, hops to a randomly chosen vacant channel. Takes
no parameters beyond the base contract. **Note:** its `channels`
parameter is not currently enforced — the vacant-channel search is not
restricted to that subset.

#### `SimpleJammer`

[![Class](https://img.shields.io/badge/Class-SimpleJammer-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_entities.md#simplejammer)
[![Dynamic](https://img.shields.io/badge/Dynamic-8884-orange?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_entities.md#simplejammer)

A naive spot-jammer with no targeting or prediction logic — each step, it
samples uniformly from whichever channels were occupied in the prior
step's ground truth.

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `avoid_repeats` | `bool` | `True` | If `True`, prefers switching to a different occupied channel over re-jamming the same one repeatedly. |

---

## `render`

Controls how the Gym environment visualizes the episode, if at all.

| Key | Type | Required | Description |
| --- | ---- | -------- | ----------- |
| `render_mode` | `str` | ✅ | Rendering backend — must be `"null"`, `"terminal"`, or `"pyqt"`. |
| `render_fps` | `int` | ✅ | Target frames per second for the renderer. |

| `render_mode` | Behavior |
| ------------- | -------- |
| `"null"` | No rendering — fastest option, recommended for training. |
| `"terminal"` | Lightweight text-based rendering to stdout. |
| `"pyqt"` | Full graphical rendering via PyQt — useful for debugging/demoing. |

```json
"render": {
  "render_mode": "pyqt",
  "render_fps": 30
}
```

---

## Full example

```json
{
  "environment": {
    "num_channels": 8,
    "max_steps": 200,
    "observation_mode": "detect",
    "reward_mode": "dsa",
    "target_entity": "incumbent_1"
  },
  "entities": {
    "incumbent_1": {
      "type": "StochasticConstantFreq",
      "channels": [3],
      "onoff": [1, 1, 0],
      "percent_on": 0.6,
      "modem_params": {
        "type": "qam", "order": 16, "filter": "RRC",
        "center_frequency": [-0.1, 0.1], "bandwidth": 0.25,
        "start": 0.25, "duration": 0.25
      }
    },
    "adversary_1": {
      "type": "SimpleJammer",
      "channels": [0, 1, 2, 3, 4, 5, 6, 7],
      "onoff": [1, 1, 0],
      "avoid_repeats": true,
      "modem_params": {
        "type": "noise", "order": 2, "filter": "RRC",
        "center_frequency": [-0.5, 0.5], "bandwidth": 1.0,
        "start": 0.0, "duration": 1.0
      }
    }
  },
  "render": {
    "render_mode": "null",
    "render_fps": 30
  }
}
```

## See also

- [![Module](https://img.shields.io/badge/Module-rfrl__gym.envs-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_envs.md) — consumes `scenario_filename` to build the simulated environment described here.
- [![Module](https://img.shields.io/badge/Module-rfrl__gym.modes-yellow?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_modes.md) — `reward_mode` here selects between `DSA` and `Jam`.
- [RFRL-Gym overview](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_intro.md) — how this scenario file feeds into the full four-layer stack.
