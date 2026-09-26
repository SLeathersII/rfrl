# `rfrl_gym.renderers`

[![Module](https://img.shields.io/badge/Module-rfrl__gym.renderers-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_renderers.md)

Renderers handle visualizing an episode as it runs — from a plain
terminal printout up to a full PyQt spectrum display. A renderer is
selected via the `render` block of a [scenario
file](./scenario_configs.md#render) and instantiated once per environment
`reset()`, not once per `step()`.

## `render_mode` options

| Value | Renderer | Description |
| ----- | -------- | ----------- |
| `"null"` | *(none)* | No rendering — fastest option, used for training. |
| `"terminal"` | `TerminalRenderer` | Lightweight text-based rendering to stdout. |
| `"pyqt"` | `PyQtRenderer` | Full graphical spectrum rendering via PyQt6. |

## Common interface

Every renderer is constructed once, then driven by two calls from the
environment:

| Method | Called from | Purpose |
| ------ | ----------- | ------- |
| `reset()` | `env.reset()` (on a hard reset) | Clears/re-initializes the renderer's internal state for a new episode. |
| `render(info)` | `env.render()`, every step | Draws the current step using the environment's live `info` dict. |

The environment paces rendering itself around `render_fps` (sleeping
between frames in `env.render()`) — the renderer's `render()` call just
draws one frame, it doesn't handle its own timing.

## `TerminalRenderer`

[![Class](https://img.shields.io/badge/Class-TerminalRenderer-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_renderers.md#terminalrenderer)

```python
TerminalRenderer(num_episodes, scenario_metadata)
```

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `num_episodes` | `int` | Total episodes the environment will run — used for progress display. |
| `scenario_metadata` | `dict` | The full parsed scenario JSON, for channel/entity context. |

## `PyQtRenderer`

[![Class](https://img.shields.io/badge/Class-PyQtRenderer-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_renderers.md#pyqtrenderer)

```python
PyQtRenderer(num_episodes, scenario_metadata, mode, samples_per_step=None)
```

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `num_episodes` | `int` | Total episodes the environment will run. |
| `scenario_metadata` | `dict` | The full parsed scenario JSON. |
| `mode` | `str` — `"abstract"` or `"iq"` | Which environment paradigm is driving it — the abstract environments pass `samples_per_step=10000` as a fixed placeholder; the IQ environments pass the real per-step sample count from `pywasp_config`. |
| `samples_per_step` | `int`, optional | IQ samples per rendered step. Required to correctly scale the spectrum display in `"iq"` mode. |

Requires a `QApplication` instance to already exist — the environment
creates one at construction (`self.pyqt_app = QApplication([])`) whenever
`render_mode == "pyqt"`, before the renderer itself is built.

## `render_history` and `render_background`

Two additional scenario-file fields appear alongside `render_mode` in
some environments but aren't part of the schema documented in [Scenario
configuration files](./scenario_configs.md) yet:

| Field | Type | Description |
| ----- | ---- | ----------- |
| `render_history` | `int` | How many past steps of spectrum data to retain for display (used to size the rolling `spectrum_data` buffer). Required and validated (`> 0`) in the IQ environments. |
| `render_background` | `str` | Display background color (e.g. `"black"`) — seen constructed in `RFRLGymAbstractEnv_LD`, not yet confirmed as a general renderer option. |

## See also

- [Scenario configuration files](./scenario_configs.md) — the `render` block that selects and configures these classes.
- [![Module](https://img.shields.io/badge/Module-rfrl__gym.envs-green?style=for-the-badge)](./api_envs.md) — every environment class constructs and drives a renderer via `render_mode`.

---

## API Reference

::: rfrl_gym.renderers
    options:
        members: true
        show_labels: true
        
