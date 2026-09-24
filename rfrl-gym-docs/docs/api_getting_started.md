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
epochs then will run inference and render in pyqt showing an agent which has learned the spectral pattern.

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
CA_CFAR sensing. IQ will be generated slower due to PyWaspGen single thread bug [Need to avoid forkbomb issue]. 

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