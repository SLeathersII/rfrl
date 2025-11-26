import gymnasium as gym
from stable_baselines3 import DQN
import rfrl_gym
from train_utils import OnlineCallbackDqn

# import matplotlib.pyplot as plt
# import json
# import os
# import time
# import random
# from train_utils import OnlineCallbackDqn
# from data_utils import Sb3_DataClass

print('packages laoded... creating environment')
# intilize environment through gym, specifying our scenario json
env = gym.make('rfrl-gym-iq-v0', scenario_filename='NAWC_test_scenario.json',
               pywasp_config = "pywaspgen/configs/default.json",
               num_episodes=3)
print('environment made')
env.reset()
print('env reset')
# print('environment reset')
# env.reset()
# print('reset twice')
# instantiate model with env
model = DQN("MlpPolicy", env, verbose=1,
            exploration_initial_eps=1.0, exploration_final_eps=0.001,
            exploration_fraction=0.995)
print('model created')
epochs = 1
# initialize online call back for rendering
ocb = OnlineCallbackDqn(render=True)
print('training model -- render:',env.unwrapped.renderer.__class__.__name__)
model = model.learn(total_timesteps=env.unwrapped.max_steps * epochs,
                    callback=ocb,
                    log_interval=100,
                    progress_bar=True)
print('done')
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