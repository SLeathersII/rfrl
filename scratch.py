import gymnasium as gym
from stable_baselines3 import DQN
from train_utils import OnlineCallbackDqn
from rfrl_gym.modes.reward_mode import *
from rfrl_gym.detectors.observation_wrappers import *
# import matplotlib.pyplot as plt
# import json
# import os
# import time
# import random
# from train_utils import OnlineCallbackDqn
# from data_utils import Sb3_DataClass

print('packages laoded... creating environment')
# intilize environment through gym, specifying our scenario json
env = gym.make('rfrl-gym-iq-v0.1', scenario_filename='sb3_test_scenario.json',
               pywasp_config = "pywaspgen/configs/default.json",
               num_episodes=3)
print('1')
env.reset()
# apply sensor and reward mode
env = OracleMap(env, 'detect')
print('sensor')
env = Jam(env)
print('reward')
# print(env.unwrapped.info['action_history']['user_agent'])
model = DQN("MlpPolicy", env, verbose=1,
            exploration_initial_eps=1.0, exploration_final_eps=0.001,
            exploration_fraction=0.995)
#
# print(env.unwrapped.info['action_history']['user_agent'])
# initialize online call back for rendering
ocb = OnlineCallbackDqn(render=True)

model = model.learn(total_timesteps=env.unwrapped.max_steps,
                    callback=ocb,
                    log_interval=100,
                    progress_bar=True)

obs, info = env.reset()
print(obs)
terminated = truncated= False
while not terminated and not truncated:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

print(env.unwrapped.info['action_history']['user_agent'])


