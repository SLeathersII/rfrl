import gymnasium as gym
import argparse
import matplotlib.pyplot as plt
import rfrl_gym
import json
import os
import time
import random
from stable_baselines3 import DQN

from train_utils import OnlineCallbackDqn
from data_utils import Sb3_DataClass

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scenario', default='NAWC_test_scenario.json', type=str,
                    help='The scenario file to preview in the RFRL gym environment.')
parser.add_argument('-k', '--kwargs', default=None, type=str, help="DQN key word arguments in 'dict'")
parser.add_argument('-r', '--render', default=True, type=bool, help="Choose if render scenario")

args = parser.parse_args()
scenario = args.scenario
scene = scenario.rsplit('.',1)[0]
env = gym.make('rfrl-gym-iq-v0', scenario_filename=scenario)
env.reset()
seed_val = random.randint(1,333333)
input_args = {"policy": "MlpPolicy",
                "env":env,
              #"learning_rate":1e-2,
                "learning_rate":0.0001,
                "buffer_size":1000000,
                "learning_starts":100,
                "batch_size":32,
                "tau":1.0,
                "gamma":0.99,
                "train_freq":4,
                "gradient_steps":1,
              "replay_buffer_class":None,
              "replay_buffer_kwargs":None,
              "optimize_memory_usage":False,
              "target_update_interval":10000,
              "exploration_fraction":0.1,
              "exploration_initial_eps":1.0,
              "exploration_final_eps":0.05,
              "max_grad_norm":10,
              "stats_window_size":100,
              "policy_kwargs":None,
              "device":'auto',
              "_init_setup_model":True,
              "verbose":0,
              "seed":seed_val,
              }
integers_set = {'learning_starts', 'buffer_size', 'batch_size', 'train_freq', 'gradient_steps',
                         'target_update_interval', 'stats_window_size','n_steps',
                         }
# convert kwargs to dict to be passed in
if args.kwargs == "None":  # fix for nested scripts
    args.kwargs = None
if args.kwargs is not None:
    kwargs = json.loads(args.kwargs)  # float kwargs MUST start with 0.xxx
    # check for None types and bool which were converted to string
    for key, value in kwargs.items():
        if value == "None":
            kwargs[key] = None
        elif value == "True" or value == "False":
            kwargs[key] = eval(value)
        elif key in integers_set:
            kwargs[key] = int(value)
    # overwrite args for use in model call
    input_args.update(kwargs)
#print(input_args)
model = DQN(**input_args)

# initialize online call back for rendering
ocb = OnlineCallbackDqn(render=args.render)

model = model.learn(total_timesteps=env.unwrapped.max_steps,
                    callback=ocb,
                    log_interval=1,
                    reset_num_timesteps = True,
                    progress_bar=False)

# save out the env data for analytics
cd = {}
cd['run_history'] = {'reward_history': model.env.buf_infos[0]['reward_history'],
                     'action_history': model.env.buf_infos[0]['action_history'],
                     'true_history': model.env.buf_infos[0]['true_history'],
                     'observation_history': model.env.buf_infos[0]['observation_history'],
                     'cumulative_reward': model.env.buf_infos[0]['cumulative_reward'][-1]}
cd['kwargs'] = input_args
del cd['kwargs']['env']  # don't want to save out env if we know scene

if ((model.env.buf_infos[0]['reward_history'] == 0).sum()/env.unwrapped.max_steps) < 0.1:
    print(f'Sb3 DQN valid run history -- scene: {scene} ... saving run')
    log = Sb3_DataClass(scene, cd)
    log.save()  # saves input cd
else:
    print('DQN bad run', scene)



