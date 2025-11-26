from abc import ABC, abstractmethod
from copy import deepcopy
import hashlib
import os
from pandas import Series
import h5py
import time


class AbstractDataClass(ABC):
    @abstractmethod
    def save(self):
        pass
    @staticmethod
    def _open(path: str,
              mode:str='r+',
              retry_limit: int = 60):
        """
        Method to wait to open h5 files and return h5 object
        :param path: path to h5 file
        :param mode: mode to open in
        :param retry_limit: time to wait for file to open
        :return: h5py.Group
        """
        attempt = 0  # attempts to open -- will be finite to avoid infinite loops
        while attempt < retry_limit:
            try:
                file = h5py.File(path, mode)
                return file
            except:  # TODO be specific (OSError IOError...)
                attempt += 1
                time.sleep(1)
        if attempt >= retry_limit:
            print(f"Unable to open file: {path}, from current working directory: {os.getcwd()}")
            return None


    @classmethod
    def _clean_kwargs(cls,
                      args:dict):
        remove = []
        for key, value in args.items():
            if value is None:
                remove.append(key)
        for key in remove:
            del args[key]

    @abstractmethod
    def _find_group(self,
                    file:h5py.Group,
                    kwargs:dict):
        """
        Checks if HyperParameter configuration has an assigned group and returns group
        assigns new group and returns that group if not found
        :param file: h5py.Group being parsed for hyperparam settings
        :param kwargs: HyperParam values being looked for

        :return: h5py.Group
        """
        pass

    def flatten_dict(self, dct, sort=False):
        """
        Flattens a nested dictionary into a single tuple containing all values.
        The order of values in the tuple depends on the dictionary's iteration order.
        """
        flat_values = []

        for key, value in dct.items():
            if isinstance(value, dict):
                flat_values.append((key, self.flatten_dict(value, sort)))
            elif isinstance(value, (tuple, list)):
                flat_values.append((key, tuple(self.flatten_dict(v, sort) if isinstance(v, dict) else v for v in value)))
            else:
                flat_values.append((key, value))
        if sort:
            flat_values.sort()
        return tuple(flat_values)

    def save_recurse(self,
                     group_data:dict,
                     h5_group:h5py.Group):
        """
        Used to recursively call dictionary groups

        :param group_data: dictionary at the input group level
        :param h5_group: current group within h5 file
        :return:
        """
        for key, value in group_data.items():
            if isinstance(value, dict):
                # create subgroup for dict and pass in recursively -- TODO this is going to make extra groups?
                subgroup = h5_group.create_group(key)
                self.save_recurse(value, subgroup)
            elif value is None:
                pass # remove None values to reduce space
            else:
                h5_group.create_dataset(key, data=value)

    def save_run(self,
                 run_data: dict,
                 h5_group:h5py.Group):
        """
        recursive function to store dictionary within a group with subgroups -- seed must be present at top level
        :param h5_group: group associated with algorithm, scene, and hyperparameter setting
        :param run_data: dictionary of numpy arrays saving out data from run
        :return: None
        """
        # use seed value to name specific run data
        seed = run_data['kwargs']['seed']
        del run_data['kwargs']['seed']

        # check if inside a group that has stored kwargs TODO make this whole logic better
        if 'run_data' in h5_group.keys():
            try:
                h5_group['run_data'].create_group(f'{seed}')
            except ValueError as e:
                print(f'seed: {seed}, error: {e}')
                return None
            for key, value in run_data.items():
                if key == "kwargs" or key =='seed':
                    pass
                elif key == 'run_history':
                    self.save_recurse(value, h5_group['run_data'][f'{seed}'])
                elif isinstance(value, dict):
                    # create subgroup for dict and pass in recursively
                    subgroup = h5_group.create_group(key)
                    self.save_recurse(value, subgroup)
                elif value is None:
                    pass  # remove None values to reduce space
                else:
                    #print(f'key:{key}, value:{value}')
                    h5_group['run_data'][f'{seed}'].create_dataset(key, data=value)

            return None # terminate input

        if 'kwargs' in h5_group.keys():
            # this should not run -- should have been terminated above if at this level
            print('something is wrong with tree structure')
            return

        # block to initialize new run group HyperParameters:x (assumes passed in at this level)
        h5_group.create_group('run_data')
        # create kwargs_hash
        kwargs_local = deepcopy(run_data['kwargs'])  # local version for checking hash
        self._clean_kwargs(kwargs_local)
        kwargs_hash = str(int(hashlib.md5(str(self.flatten_dict(kwargs_local, True)).encode('utf-8')).hexdigest(), 16))
        h5_group.create_dataset('kwargs_hash', data=kwargs_hash)

        print('new kwargs_hash: ', kwargs_hash, kwargs_local)
        for key, value in run_data.items():
            if key == 'run_history':
                h5_group['run_data'].create_group(f'{seed}')
                self.save_recurse(value, h5_group['run_data'][f'{seed}'])
            elif isinstance(value, dict):
                # create subgroup for dict and pass in recursively --
                subgroup = h5_group.create_group(key)
                self.save_recurse(value, subgroup)
            elif value is None:
                pass # remove None values to reduce space
            elif key == 'seed':
                pass
                #h5_group['run_data'][f'{seed}'].create_dataset(key, data=value)
            else:  # this is bad practice
                h5_group.create_dataset(key, data=value)

        return None

    def load_runs(self,
                  path: str):
        pass

    @abstractmethod
    def _init__setup(self):
        pass
    @abstractmethod
    def h5_group_to_dict(self,
                         h5_group:h5py.Group):
        """
        Method to help parse metadata settings in saved file -- check group against algorithm kwargs
        :param h5_group:
        :return:
        """
        pass
    @staticmethod
    def reward_moving_average(run_series:Series,
                              window:int = 25,
                              ):
        if isinstance(run_series, Series):
            return run_series.rolling(window).mean().dropna()
        else:
            run_series = Series(run_series)
            return run_series.rolling(window).mean().dropna()