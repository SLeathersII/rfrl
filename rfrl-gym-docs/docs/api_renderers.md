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
        
# `rfrl_gym.renderers`

[![Module](https://img.shields.io/badge/Module-rfrl__gym.renderers-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_renderers.md)

Renderers visualize an episode as it runs — from a plain terminal
printout up to a full PyQt spectrum display. A renderer is selected via
the `render` block of a [scenario file](./scenario_configs.md#render),
constructed once per environment `reset()`, and driven once per
`env.render()` call thereafter.

## `render_mode` options

| Value | Renderer | Description |
| ----- | -------- | ----------- |
| `"null"` | *(none)* | No rendering — fastest, used for training. |
| `"terminal"` | `TerminalRenderer` | Text-based channel-occupancy grid printed to stdout. |
| `"pyqt"` | `PyQtRenderer` | Full graphical dashboard via PyQt6/pyqtgraph. |

## `Renderer` — shared base class

[![Class](https://img.shields.io/badge/Class-Renderer-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_renderers.md#renderer)

Both concrete renderers subclass `Renderer`, which handles construction
and defines the two-method public contract every renderer exposes:

```python
Renderer(num_episodes, scenario_metadata)
```

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `num_episodes` | `int` | Total episodes the environment will run. |
| `scenario_metadata` | `dict` | The full parsed scenario JSON — used to pull `num_channels`, `observation_mode`, `render_history`, and the entity list. |

| Attribute | Source | Description |
| --------- | ------ | ----------- |
| `num_channels` | `scenario_metadata['environment']['num_channels']` | Channel count. |
| `observation_mode` | `scenario_metadata['environment']['observation_mode']` | `'detect'` or `'classify'` — changes what symbols/colors each renderer draws. |
| `render_history` | `scenario_metadata['render']['render_history']` | How many past steps to keep visible in the rolling display. |
| `entity_list` | `scenario_metadata['entities']` | Raw entity dict from the scenario file (not instantiated `Entity` objects). |
| `num_entities` | `len(scenario_metadata['entities'])` | Entity count. |

| Method | Delegates to | Called from |
| ------ | ------------- | ----------- |
| `render(info)` | `self._render()` | `env.render()`, every step — stores `info` on `self` first, then calls the subclass hook. |
| `reset()` | `self._reset()` | `env.reset()`, on a hard reset. |

`_render()` and `_reset()` are the two hooks every subclass implements —
`Renderer` itself defines the public interface but no drawing logic.

## `TerminalRenderer`

[![Class](https://img.shields.io/badge/Class-TerminalRenderer-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_renderers.md#terminalrenderer)

```python
TerminalRenderer(num_episodes, scenario_metadata)
```

Same constructor as the base class — no additional parameters. Each
`_render()` call prints three sections, bounded above and below by a
`=`-divider sized to `render_history`:

1. **Occupancy grid** — one row per channel, one column per step (most
   recent `render_history` steps, newest first). Each cell's symbol
   depends on `observation_mode` and whether the player's action matches
   that channel:

   | Observation mode | Cell meaning | Symbol |
   | ---- | ------------ | ------ |
   | `detect` | Player on an empty channel | `1` |
   | `detect` | Player collided with an entity | `X` |
   | `detect` | Entity present, player elsewhere | `O` |
   | `classify` | Player on an empty channel | `0` |
   | `classify` | Player collided with entity *N* | `xNx` |
   | `classify` | Player collided with multiple entities | `xCx` |
   | `classify` | Entity *N* present, player elsewhere | `N` |
   | `classify` | Multiple entities collided (no player) | `C` |

2. **Step numbers** — a zero-padded row of step indices aligned under
   the occupancy grid.
3. **Reward summary** — current step reward, cumulative reward, and (if
   `num_episodes > 1`) the full episode-reward history, followed by a
   legend matching the symbol table above.

`_reset()` is a no-op — this renderer has no internal state to clear.

## `PyQtRenderer`

[![Class](https://img.shields.io/badge/Class-PyQtRenderer-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_renderers.md#pyqtrenderer)

```python
PyQtRenderer(num_episodes, scenario_metadata, mode, samples_per_step)
```

Subclasses both `Renderer` and `QMainWindow`. Builds a fixed
1400×900 dashboard window at construction, with layout that branches on
`mode`:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `mode` | `str` — `"abstract"` or `"iq"` | `"abstract"` shows only the occupancy map; `"iq"` additionally adds tabbed **Spectrum View** and **Sensing View** panels. |
| `samples_per_step` | `int` | IQ samples per step — sets up the time/frequency axes and the low-pass filter (`scipy.signal.butter`, order 30) used for per-channel energy sensing. |

Also reads `scenario_metadata['render']['render_background']` (`'white'`
or `'black'`) to set the whole window's color scheme, including plot
pens, grid colors, and the legend's text color.

### Panels

| Panel | Always shown? | Draws |
| ----- | -------------- | ----- |
| Occupancy map | Always | Same channel × step grid as `TerminalRenderer`, as a scrolling color image (green = player on empty channel, red = player collision, blue = entity present in `detect` mode, per-entity distinct colors via `distinctipy` in `classify` mode, yellow = multi-entity collision). |
| Spectrum View | `mode == 'iq'` only | A spectrogram (`scipy.signal.spectrogram`, `nperseg=512`) of `info['spectrum_data']`, in dB, frequency-shifted to center DC. |
| Sensing View | `mode == 'iq'` only | Per-channel energy: each channel is downmixed to baseband (`exp(-1j·2π·fc[k]·t)`), low-pass filtered, downsampled, and summed as `|x|²` — a simple energy detector visualization, independent of whatever detector wrapper is actually attached to the environment. |
| Cumulative Reward per Step | Always | Line plot of `info['cumulative_reward']` up to the current step. |
| Episode Rewards | Only if `num_episodes != 1` | Line plot of `info['episode_reward']` across completed episodes. |
| Legend | Always | Color key — entity list and collision colors, matching the occupancy map's palette. |

`_render()` redraws the occupancy image (and, in `iq` mode, the
spectrum/sensing images) every call, rescales axis ticks to the current
step, updates both reward plots, and calls `QApplication.processEvents()`
to flush the redraw. On the very first call it also shows the window and
applies plot padding.

`_reset()` clears the occupancy and sensing image buffers and the
cumulative-reward plot back to empty/zero.

> **Dependency note:** requires `distinctipy`, `pyqtgraph`, and a
> `logowhite.png` file in the working directory (loaded via
> `matplotlib.pyplot.imread` for the logo panel) — none of which are
> needed for `TerminalRenderer` or `render_mode: "null"`.

## `render` schema fields these renderers consume

| Field | Type | Used by | Description |
| ----- | ---- | ------- | ----------- |
| `render_mode` | `str` | Environment | Selects which renderer (if any) gets constructed. |
| `render_fps` | `int` | Environment | Paces the render loop between `render()` calls (not read by the renderer classes themselves). |
| `render_history` | `int` | Both renderers | Width of the rolling occupancy/spectrum display, in steps. |
| `render_background` | `str` — `"white"` or `"black"` | `PyQtRenderer` only | Window color scheme. Not read by `TerminalRenderer`. |

## See also

- [Scenario configuration files](./scenario_configs.md) — the `render` block that selects and configures these classes.
- [![Module](https://img.shields.io/badge/Module-rfrl__gym.envs-green?style=for-the-badge)](./api_envs.md) — every environment constructs and drives a renderer via `render_mode`.

---

## API Reference

::: rfrl_gym.renderers
    options:
        members: true
        show_labels: true
