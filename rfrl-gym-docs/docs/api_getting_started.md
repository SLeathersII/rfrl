# Getting Started

[![Guide](https://img.shields.io/badge/Docs-Getting%20Started-blue?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_getting_started.md)

This page walks through installing RFRL-Gym and running its two example
training scripts — one showing the original abstract-gym workflow, one
showing the current IQ-based pipeline with sensing and reward wrappers.

## Install and build

```bash
# 1. Clone the repository
git clone https://github.com/SLeathersII/rfrl
cd rfrl

# 2. Initialize a uv project in the existing folder
uv sync

# 3. Run a script with uv run
uv run rfrl-gym/scripts/sb3_example.py
```

---

## Example 1 — `sb3_example.py` (abstract gym, legacy path)

[![Env](https://img.shields.io/badge/Env-rfrl--gym--abstract--v0-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_envs.md)
[![Mode](https://img.shields.io/badge/Mode-Jam-yellow?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_modes.md#jam)

Trains and then demonstrates a jamming agent on the original monolithic
`rfrl-gym-abstract-v0` environment — no separate detector/reward wrappers,
since this predates that split.

| Step | What happens |
| ---- | ------------ |
| 1. Parse CLI args | `--scenario` (default `sb3_test_scenario.json`), `--gym_mode` (`abstract` or `iq`), `--epochs` (default `200`). |
| 2. Build the env | `gym.make('rfrl-gym-abstract-v0', scenario_filename=...)` — over 10 channels, in jam mode, per the default scenario file. |
| 3. Train | A Stable-Baselines3 `DQN` (`MlpPolicy`) trains for `max_steps * epochs` total timesteps, with epsilon-greedy exploration annealed from `1.0` down to `0.001` over 99.5% of training. |
| 4. Save & reload | The trained model is saved to disk (`rfrl_gym_dqn`), deleted from memory, then reloaded from that file — a deliberate round-trip to demonstrate save/load works, not a required step in normal use. |
| 5. Run inference | The reloaded model runs one full deterministic episode (`model.predict(..., deterministic=True)`), calling `env.render()` each step. |

**Result:** after ~200 epochs of training, you get a PyQt-rendered episode
showing an agent that has learned a targeted jamming policy.

**Run it:**
```bash
uv run rfrl-gym/scripts/sb3_example.py --epochs 200 --gym_mode abstract
```

---

## Example 2 — `scratch_train.py` (current IQ pipeline)

[![Env](https://img.shields.io/badge/Env-rfrl--gym--iq--v0.1-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_envs.md)
[![Detector](https://img.shields.io/badge/Detector-CA__CFAR-yellow?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_detectors.md)
[![Mode](https://img.shields.io/badge/Mode-DSA-green?style=for-the-badge)](https://github.com/SLeathersII/rfrl/blob/main/rfrl-gym-docs/docs/api_modes.md#dsa)

This is the modern, fully-layered version: real IQ simulation, an actual
CFAR-based sensing wrapper instead of oracle ground truth, and the DSA
reward objective — i.e. this is `envs → detectors → modes → agent` in
practice, not just in the diagram.

> ⚠️ Runs slower than Example 1 — PyWaspGen currently has a single-thread
> bottleneck, and CFAR sensing adds real per-step computation on top of it.

| Step | What happens |
| ---- | ------------ |
| 1. Parse CLI args | `--scenario`, `--pywaspgen_config` (default `pywaspgen/configs/default.json`), `--gym_mode` (default `rfrl-gym-iq-v0.1`), `--epochs` (default `5` — much lower than Example 1, reflecting the heavier per-step cost). |
| 2. Build the base env | `gym.make(gym_mode, scenario_filename=..., pywasp_config=..., num_episodes=1)`. |
| 3. Wrap with a detector | `CA_CFAR(env, 'detect', p_fa=0.01)` — Cell-Averaging Constant False Alarm Rate detection, with a 1% false-alarm probability target. The script's comments show this slot is interchangeable: `EnergyDetector` and `OracleMap` are alternative detectors that can drop in here unmodified. |
| 4. Wrap with a reward mode | `DSA(env)` — dynamic spectrum access reward, layered on top of the detector from step 3. (Commented-out `Jam(env)` shows this is swappable too.) |
| 5. Train | Same `DQN`/`MlpPolicy` setup as Example 1, but only `max_steps * epochs` timesteps (5 epochs by default) and a shorter exploration-annealing fraction (`0.9` vs `0.995`). |
| 6. (Optional) online callback | An `OnlineCallbackDqn` hook exists for live rendering during training itself, but is disabled (`ocb = None`) by default — set it to render as the agent learns, not just at inference time. |
| 7. Run inference | Same deterministic rollout-and-render loop as Example 1 — no save/load round-trip this time. |

**The key structural difference from Example 1:** steps 3–4 are exactly
the wrapper-composition pattern from the [architecture
overview](./api_intro.md) — swap `CA_CFAR` for `EnergyDetector` or
`OracleMap`, or `DSA` for `Jam`, and nothing else in the script needs to
change.

**Run it:**
```bash
uv run scratch_train.py --epochs 5
```

---

## Comparing the two examples

| | `sb3_example.py` | `scratch_train.py` |
| --- | --- | --- |
| Environment | `rfrl-gym-abstract-v0` | `rfrl-gym-iq-v0.1` |
| Sensing | None (abstract state) | `CA_CFAR` (swappable) |
| Reward | Jam (hardcoded) | `DSA` (swappable) |
| Default epochs | 200 | 5 |
| Exploration anneal fraction | 0.995 | 0.9 |
| Save/load demo | Yes | No |
| Speed | Fast | Slower (PyWaspGen + CFAR overhead) |

If you're learning the framework, start with Example 1 to see the
end-to-end training/inference loop without wrapper complexity, then move
to Example 2 to see how detector and mode wrappers actually compose in a
real training script.

## See also

- [RFRL-Gym architecture overview](./api_intro.md) — the four-layer stack these scripts assemble.
- [![Module](https://img.shields.io/badge/Module-rfrl__gym.detectors-yellow?style=for-the-badge)](./api_detectors.md) — full list of detector wrappers, including `CA_CFAR`, `EnergyDetector`, `OracleMap`.
- [![Module](https://img.shields.io/badge/Module-rfrl__gym.modes-yellow?style=for-the-badge)](./api_modes.md) — `DSA` and `Jam` reward-mode details.
- [Scenario configuration files](./scenario_configs.md) — the JSON schema behind `--scenario`.


# Install and Build RFRL-Gym
```bash
# 1. Clone the repository for local scripts
git clone https://github.com/SLeathersII/rfrl
cd rfrl

# 2. Initialize a uv project structure in the existing folder
uv sync

# 3. Run the package or script using uv run
uv run rfrl-gym/scripts/sb3_example.py
```
This will make one of the original monolithic abstract gyms ('rfrl-gym-abstract-v0') in jam mode over 10 channels. It will first train for 200
epochs then will run inference and render in pyqt showing an agent which has learned a targeted jamming policy.

```python
import gymnasium as gym
import argparse
import rfrl_gym
from stable_baselines3 import DQN

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scenario', default='sb3_test_scenario.json',
                    type=str, help='The scenario file to preview in the RFRL gym environment.')
parser.add_argument('-m', '--gym_mode', default='abstract',
                    type=str, help='Which type of RFRL gym environment to run.')
parser.add_argument('-e', '--epochs', default=200,
                    type=int, help='Number of training epochs.')
args = parser.parse_args()

if args.gym_mode == 'abstract':
    env = gym.make('rfrl-gym-abstract-v0', scenario_filename=args.scenario)
elif args.gym_mode == 'iq':
    env = gym.make('rfrl-gym-iq-v0.1', scenario_filename=args.scenario)
env.reset()

model = DQN("MlpPolicy", env, verbose=1, exploration_initial_eps=1.0,
            exploration_final_eps=0.001,exploration_fraction=0.995)
model = model.learn(total_timesteps=env.unwrapped.max_steps * args.epochs,
                    log_interval=100, progress_bar=True)
env.reset()
model.save("rfrl_gym_dqn")

del model # remove to demonstrate saving and loading

model = DQN.load("rfrl_gym_dqn")


obs, info = env.reset()
terminated = truncated= False
running_reward = 0
rewards = []

while not terminated and not truncated:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
```
For modern functionality run 

```bash
uv run scratch_train.py
```
which will use the RFRL gym updates on the same scenario to convert the reward function to DSA with a wrapper and 
CA_CFAR sensing. Gym will run slower due to PyWaspGen single thread bug AND CFAR sensing computation. 

```python
import gymnasium as gym
import rfrl_gym
import argparse
from stable_baselines3 import DQN
from train_utils import OnlineCallbackDqn
from rfrl_gym.detectors.observation_wrappers import *
from rfrl_gym.modes.reward_mode import *


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scenario', default='sb3_test_scenario.json',
                    type=str, help='The scenario file to preview in the RFRL gym environment.')
parser.add_argument('-c', '--pywaspgen_config', default="pywaspgen/configs/default.json",
                    type=str, help='The scenario file to preview in the RFRL gym environment.')
parser.add_argument('-m', '--gym_mode', default='rfrl-gym-iq-v0.1',
                    type=str, help='Which type of RFRL gym environment to run.')
parser.add_argument('-e', '--epochs', default=5,
                    type=int, help='Number of training epochs.')
args = parser.parse_args()

env = gym.make(args.gym_mode, scenario_filename=args.scenario,
               pywasp_config =args.pywaspgen_config,
               num_episodes=1)
# Apply sensor
env = CA_CFAR(env, 'detect', p_fa=0.01)
                #env = EnergyDetector(env, 'detect')
                #env = OracleMap(env, 'detect')
                #env = Jam(env)
# Apply reward wrapper
env = DSA(env)
env.reset()
# Online Call Back for online learning and rendering
#ocb = OnlineCallbackDqn(render=True)
ocb = None
model = DQN("MlpPolicy", env,
            verbose=1, exploration_initial_eps=1.0,
            exploration_final_eps=0.001,exploration_fraction=0.9)

model = model.learn(total_timesteps=env.unwrapped.max_steps*args.epochs, # epochs
                    callback=ocb,
                    log_interval=100,
                    progress_bar=True)

obs, info = env.reset()
terminated = truncated= False
running_reward = 0
rewards = []

while not terminated and not truncated:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
```
