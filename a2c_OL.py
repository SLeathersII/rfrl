import gymnasium as gym
import argparse
import json
import random
from stable_baselines3 import A2C
from train_utils import OnlineCallback
from data_utils import Sb3_DataClass

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scenario', default='NAWC_test_scenario.json', type=str,
                    help='The scenario file to preview in the RFRL gym environment.')
parser.add_argument('-k', '--kwargs', default=None, type=str, help="A2C key word arguments in 'dict'")
parser.add_argument('-r', '--render', default=False, type=bool, help="Choose if render scenario")

args = parser.parse_args()
scenario = args.scenario
scene = scenario.rsplit('.',1)[0]
env = gym.make('rfrl-gym-iq-v0', scenario_filename=scenario)
env.reset()
seed_val = random.randint(1,333333)
input_args = {"policy": "MlpPolicy",
                "env":env,
                "learning_rate":0.0007,
                "n_steps":1,
                "gamma":0.99,
                "gae_lambda":1.0,
                "ent_coef":0.0,
                "vf_coef":0.5,
                "max_grad_norm":0.5,
                "rms_prop_eps":1e-5,
                "use_rms_prop":True,
              "use_sde":False,
              "sde_sample_freq":-1,
              "rollout_buffer_class":None,
              "rollout_buffer_kwargs":None,
              "normalize_advantage":False,
              "stats_window_size": 100,
              "policy_kwargs":None,
              "verbose":0,
              "seed":seed_val,
              "device":'auto',
              }
integers_set = {'learning_starts', 'buffer_size', 'batch_size', 'train_freq', 'gradient_steps',
                         'target_update_interval','stats_window_size','n_steps',
                         }
# convert kwargs to dict to be passed in
if args.kwargs == "None":  # fix for nested scripts
    args.kwargs = None
if args.kwargs is not None:
    kwargs = json.loads(args.kwargs)  # float kwargs MUST start with 0.xxx
    for key, value in kwargs.items():
        if value == "None":
            kwargs[key] = None
        elif value == "True" or value == "False":
            kwargs[key] = eval(value)
        elif key in integers_set:
            kwargs[key] = int(value)
    # overwrite args for use in model call
    input_args.update(kwargs)

model = A2C(**input_args)

# initialize online call back for rendering
ocb = OnlineCallback(render=args.render)

model = model.learn(total_timesteps=env.unwrapped.max_steps,
                    callback=ocb,
                    log_interval=1,
                    progress_bar=False)

# save out the env data for analytics (env saved in ocb)
cd = {}
cd['run_history'] = {'reward_history': model.env.buf_infos[0]['reward_history'],
                     'action_history': model.env.buf_infos[0]['action_history'],
                     'true_history': model.env.buf_infos[0]['true_history'],
                     'observation_history': model.env.buf_infos[0]['observation_history'],
                     'cumulative_reward': model.env.buf_infos[0]['cumulative_reward'][-1]}
cd['kwargs'] = input_args
del cd['kwargs']['env']  # don't want to save out env if we know scene


if ((model.env.buf_infos[0]['reward_history'] == 0).sum()/env.unwrapped.max_steps) < 0.1:
    print(f'Sb3 A2C valid run history -- scene: {scene} ... saving run')
    log = Sb3_DataClass(scene, cd, algorithm='SB3_A2C')
    log.save()
