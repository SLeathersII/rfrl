import gymnasium as gym
from stable_baselines3 import DQN
import rfrl_gym
from train_utils import OnlineCallbackDqn
import numpy as np
import matplotlib.pyplot as plt
from rfrl_gym.detectors.observation_wrappers import *
from rfrl_gym.modes.reward_mode import *
import pywaspgen
from stable_baselines3 import DQN
from train_utils import OnlineCallbackDqn

# import matplotlib.pyplot as plt
# import json
# import os
# import time
# import random
# from train_utils import OnlineCallbackDqn
# from data_utils import Sb3_DataClass

# intilize environment through gym, specifying our scenario json
# env = gym.make('rfrl-gym-iq-v0.1', scenario_filename='NAWC_test_scenario.json',
#                pywasp_config = "pywaspgen/configs/default.json",
#                num_episodes=10)
env = gym.make('rfrl-gym-wild-iq-v0', scenario_filename='NAWC_test_scenario.json',
               pywasp_config = "pywaspgen/configs/default.json",
               num_episodes=10)

env.reset() # intialize info
env = EnergyDetector(env, 'detect') # apply sensor
#env = DSA(env) # apply reward mode
# env = OracleMap(env, 'detect')
#env = Jam(env)
env = DSA(env)
env.reset()

ocb = OnlineCallbackDqn(render=True)
model = DQN("MlpPolicy", env,
            verbose=1, exploration_initial_eps=1.0,
            exploration_final_eps=0.001,exploration_fraction=0.9)

model = model.learn(total_timesteps=env.unwrapped.max_steps,
                    callback=ocb,
                    log_interval=100,
                    progress_bar=True)

# obs, info = env.reset()
# terminated = truncated= False
# running_reward = 0
# rewards = []
#
# while not terminated and not truncated:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     env.render()

# # plot 1
# iq_gen = pywaspgen.IQDatagen("pywaspgen/configs/default.json")
# burst_gen = pywaspgen.BurstDatagen("pywaspgen/configs/default.json")
# iq_data, updated_burst_list = iq_gen.gen_iqdata(env.unwrapped.user_burst_list)
# iq_gen.plot_iqdata(iq_data[0])
# plt.hlines(y=np.linspace(-.5, 0.496, 10+1), xmin=0, xmax=10_000)
# plt.show(block=False)
# # plot 1 done
# s = EnergyDetector(env, 'detect')
# # plot 2 wrapped
# iq_gen = pywaspgen.IQDatagen("pywaspgen/configs/default.json")
# burst_gen = pywaspgen.BurstDatagen("pywaspgen/configs/default.json")
# iq_data, updated_burst_list = iq_gen.gen_iqdata(s.unwrapped.user_burst_list)
# iq_gen.plot_iqdata(iq_data[0])
# plt.hlines(y=np.linspace(-.5, 0.496, 10+1), xmin=0, xmax=10_000)
# plt.show(block=False)
# # plot 2 wrapped
# obs, reward, term, trunc, info = s.step(1)
# # plot 3 after step
# iq_gen = pywaspgen.IQDatagen("pywaspgen/configs/default.json")
# burst_gen = pywaspgen.BurstDatagen("pywaspgen/configs/default.json")
# iq_data, updated_burst_list = iq_gen.gen_iqdata(s.unwrapped.user_burst_list)
# iq_gen.plot_iqdata(iq_data[0])
# plt.hlines(y=np.linspace(-.5, 0.496, 10+1), xmin=0, xmax=10_000)
# plt.show(block=False)
# # plot 3 done
# print(f'obs: {obs}\nsensing_energy_history: {info["sensing_energy_history"][0:3]}\ntrue_history: {info["true_history"][0:3]})')
# s.render()
# env = gym.make('rfrl-gym-iq-v0', scenario_filename='NAWC_test_scenario.json',
#                pywasp_config = "pywaspgen/configs/default.json",
#                num_episodes=10)
# # env.reset()
# # obs, reward, term, trunc, info = env.step(1)
# # env.render()
# #
# # print('pywasp branch', "-"*30)
# # print(f'obs: {obs}\nsensing_energy_history: {info["sensing_energy_history"][0:3]}\ntrue_history: {info["true_history"][0:3]})')
#
# # blocking
# plt.show()









# print('packages laoded... creating environment')
# # intilize environment through gym, specifying our scenario json
# env = gym.make('rfrl-gym-iq-v0', scenario_filename='NAWC_test_scenario.json',
#                pywasp_config = "pywaspgen/configs/default.json",
#                num_episodes=3)
# print('environment made')
# env.reset()
# print('env reset')
# # print('environment reset')
# # env.reset()
# # print('reset twice')
# # instantiate model with env
# model = DQN("MlpPolicy", env, verbose=1,
#             exploration_initial_eps=1.0, exploration_final_eps=0.001,
#             exploration_fraction=0.995)
# print('model created')
# epochs = 1
# # initialize online call back for rendering
# ocb = OnlineCallbackDqn(render=True)
# print('training model -- render:',env.unwrapped.renderer.__class__.__name__)
# model = model.learn(total_timesteps=env.unwrapped.max_steps * epochs,
#                     callback=ocb,
#                     log_interval=100,
#                     progress_bar=True)
# print('done')
# print(env.unwrapped.info['sensing_energy_history'])

# obs, info = env.reset()
# terminated = truncated= False
# running_reward = 0
# rewards = []
# print('running model')
# while not terminated and not truncated:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     env.render()
#     print(reward)