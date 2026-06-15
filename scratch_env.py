import gymnasium as gym
from stable_baselines3 import DQN
import rfrl_gym
from train_utils import OnlineCallbackDqn
import numpy as np
import matplotlib.pyplot as plt
from rfrl_gym.detectors.observation_wrappers import EnergyDetector
from rfrl_gym.modes.reward_mode import DSA
import pywaspgen
from stable_baselines3 import DQN

env = gym.make('rfrl-gym-iq-v0.1', scenario_filename='NAWC_test_scenario.json',
               pywasp_config = "pywaspgen/configs/default.json",
               num_episodes=10)
env.reset() # intialize info

env = EnergyDetector(env, 'detect') # apply sensor
env.reset()
obs, reward, term, trunc, info = env.step(1)
print(f'obs: {obs}\nsensing_energy_history: {info["sensing_energy_history"][0:3]}\ntrue_history: {info["true_history"][0:3]}')
print(f'observation_history: {info['observation_history'][0:3]}')
env = DSA(env) # apply reward mode
env.reset()

# iq_gen = pywaspgen.IQDatagen("pywaspgen/configs/default.json")
# burst_gen = pywaspgen.BurstDatagen("pywaspgen/configs/default.json")
# iq_data, updated_burst_list = iq_gen.gen_iqdata(env.unwrapped.user_burst_list)
# iq_gen.plot_iqdata(iq_data[0])
# plt.hlines(y=np.linspace(-.5, 0.496, 10+1), xmin=0, xmax=10_000)
# plt.show(block=False)

obs, reward, term, trunc, info = env.step(1)
# # env.render()


print(f'obs: {obs}\nsensing_energy_history: {info["sensing_energy_history"][0:3]}\ntrue_history: {info["true_history"][0:3]}')
print(f'observation_history: {info['observation_history'][0:3]}')