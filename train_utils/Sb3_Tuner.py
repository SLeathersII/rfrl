from scipy.stats import qmc
from copy import deepcopy
import numpy as np
import subprocess
import os


class Tuner:

    def __init__(self,
                 study_params: int = 3,
                 study_dict: dict = None,
                 num_runs: int = None,
                 algorithm: str = 'DQN',
                 seed: int = 3,
                 ):
        """
        study_params: int defining number of HPs you are tuning -- must match input tuples of study_dict
        study_dict: a dict of HPs keys with tuple values with (lower, upper) bound on each HP->
            {key1: (lower1, upper1), key2: (lower2, upper2)}... keys MUST match keywords
            for boolean valued hyperparameters uses (0,1) for tuple bounds
        num_runs: int determining how many samples you would like generated -- if not set will use 10 * study_params
        NOTE: This format assumes you are tuning Sb3 models
        """
        self.study_params = study_params
        if not num_runs:
            self.samples = 10 * self.study_params
        else:
            self.samples = num_runs
        # default dictionary to check against/use **kwargs -- Maintain default values here -- must add new Algos
        if algorithm.lower() == "dqn":
            self.default_study_dict = {"policy": "MlpPolicy",
                                       "learning_rate": 0.0001,
                                       "buffer_size": 1000000,
                                       "learning_starts": 100,
                                       "batch_size": 32,
                                       "tau": 1.0,
                                       "gamma": 0.99,
                                       "train_freq": 4,
                                       "gradient_steps": 1,
                                       "replay_buffer_class": None,
                                       "replay_buffer_kwargs": None,
                                       "optimize_memory_usage": False,
                                       "target_update_interval": 10000,
                                       "exploration_fraction": 0.1,
                                       "exploration_initial_eps": 1.0,
                                       "exploration_final_eps": 0.05,
                                       "max_grad_norm": 10,
                                       "stats_window_size": 100,
                                       "policy_kwargs": None,
                                       "device": 'auto',
                                       "_init_setup_model": True,
                                       "verbose": 0,
                            }
        elif algorithm.lower() == 'a2c':
            self.default_study_dict =  {"policy": "MlpPolicy",
                                        "learning_rate": 0.0007,
                                        "n_steps": 1,  # needed to pull out information correctly
                                        "gamma": 0.99,
                                        "gae_lambda": 1.0,
                                        "ent_coef": 0.0,
                                        "vf_coef": 0.5,
                                        "max_grad_norm": 0.5,
                                        "rms_prop_eps": 1e-5,
                                        "use_rms_prop": True,
                                        "use_sde": False,
                                        "sde_sample_freq": -1,
                                        "rollout_buffer_class": None,
                                        "rollout_buffer_kwargs": None,
                                        "normalize_advantage": False,
                                        "stats_window_size": 100,
                                        "policy_kwargs": None,
                                        "verbose": 0,
                                        "device": 'auto',
                                        }
        self.integers = {'learning_starts', 'buffer_size', 'batch_size', 'train_freq', 'gradient_steps',
                         'target_update_interval', 'stats_window_size','n_steps', 'verbose', 'max_grad_norm'
                         }
        study = deepcopy(self.default_study_dict)
        if study_dict is not None:
            study.update(study_dict)
        self.study_dict = study
        # cheap check to insure no new keywords have been introduced
        assert len(self.study_dict) == len(self.default_study_dict)
        self.seed = seed # want to avoid putting seed in study dict for later logic
        self.study_samples = self.study(seed = self.seed)
        self.algorithm = algorithm

    def study(self,
              seed: int = None,
              ):

        lhc = qmc.LatinHypercube(self.study_params, seed=seed)
        study_samples = lhc.random(self.samples)
        # find the values in our study_dict with tuples to define ranges
        upper = []
        lower = []
        for key, value in self.study_dict.items():
            if isinstance(value, (bool, type(None), str)):
                self.study_dict[key] = np.array([value] * self.samples)
            elif isinstance(value, tuple):
                lower.append(value[0])  # unpack tuple values
                upper.append(value[1])
            elif isinstance(value, (int, float)):
                self.study_dict[key] = np.zeros(self.samples) + value
        if (len(upper) + len(lower)) > 0:
            study_samples = qmc.scale(study_samples, lower, upper)
            # update ranges with study values
            count = 0  # counter for each study value to maintain order in dict
            for key, value in self.study_dict.items():
                if type(value) == tuple:
                    if type(self.default_study_dict[key]) == bool:
                        self.study_dict[key] = study_samples[0:, count] > .5
                    elif key in self.integers:
                        self.study_dict[key] = study_samples[0:, count].astype(int)
                    else:
                        self.study_dict[key] = study_samples[0:, count]
                    count += 1
        # type fix
        for key, value in self.study_dict.items():
            if key in self.integers:
                self.study_dict[key] = self.study_dict[key].astype(int)

        return study_samples

    def run_study(self,
                  scenes: tuple[str,...] = ('NAWC_static_3.json','NAWC_static_4.json','NAWC_static_5.json'),
                  workers: int = 8,
                  runs_per_hp: int = 30):
        """
        Run Sb3 model of algorithm used at initialization
        :param scenes: scenes to test hyperparameters on
        :param workers: Number of threads to use/ simultaneous sims and agents
        :param runs_per_hp: for each hyper parameter setting, the number of seeds run at that setting
        :return:
        """
        # TODO set up command structure such that it will wrap to new scene or hp_set to use
        #  full number of workers at all times

        #path logic for calling script in same folder
        path_original = os.getcwd()
        path_desired = os.path.abspath(__file__)[:-25]
        os.chdir(path_desired)
        print(f'cwd: {os.getcwd()}')
        val_wrap = lambda key, value, run: f'"{key}":{value[run]}'
        string_wrap = lambda key, value, run: f'"{key}":"{value[run]}"'
        for scene in scenes:
            for run_per_hp in range(runs_per_hp):
                run = 0
                while run < self.samples:
                    command_string = ''
                    for worker in range(workers):
                        if self.algorithm.lower() == 'dqn':
                            base_model_call = f'python dqn_OL.py --scenario {scene} --kwargs '
                        elif self.algorithm.lower() == 'a2c':
                            base_model_call = f'python a2c_OL.py --scenario {scene} --kwargs '
                        kwargs = "'{"
                        for idx, (key, value) in enumerate(self.study_dict.items()):
                            if isinstance(value[run], (type(None), np.bool_, np.str_)):
                                kwargs += string_wrap(key, value, run)
                            else:
                                kwargs += val_wrap(key, value, run)
                            if idx + 1 != len(self.study_dict):
                                kwargs += ','
                        kwargs += "}'"
                        command_string+= base_model_call + kwargs
                        run+= 1
                        if run >= self.samples:
                            break  # don't want to index out of samples
                        if worker + 1 != workers:
                            command_string+= " & "  # should only add if looping back
                    subprocess.run(command_string, shell = True)
        os.chdir(path_original)
