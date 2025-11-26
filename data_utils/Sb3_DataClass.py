import h5py
import os
import hashlib
from copy import deepcopy


from .AbstractDataClass import AbstractDataClass

class Sb3DataClass(AbstractDataClass):

    def __init__(self,
                 scene: str,
                 run_data:dict=None,
                 algorithm:str = 'SB3_DQN',
                 data_path:str = '../experiments'
                 ):
        assert type(run_data) is dict, "run data must be input {'kwargs':{}, 'data':[]...}"
        # no default kwargs, must be input and map to algorithm
        self.scene = scene
        self.algorithm = algorithm
        self.run_data = run_data
        self.data_path = data_path
        self._init__setup()

    def _init__setup(self):
        self.file = f'{self.data_path}/{self.algorithm}/{self.scene}/run_data.h5'
        os.makedirs(self.file[:-12], exist_ok=True)

    def _find_group(self,
                    file:h5py.Group,
                    kwargs:dict):
        """
        Checks if HyperParameter configuration has an assigned group and returns group
        :param file: h5py.Group being parsed for hyperparam settings
        :param kwargs: HyperParam values being looked for

        :return: h5py.Group or None
        """
        # helper function to clean out None values
        self._clean_kwargs(kwargs)
        kwargs_l = deepcopy(kwargs) # local version for checking hash
        del kwargs_l['seed']
        #print('kwargs_l: ', kwargs_l)
        kwargs_hash = str(int(hashlib.md5(str(self.flatten_dict(kwargs_l, True)).encode('utf-8')).hexdigest(), 16))
        print(f'kwargs_hash: {kwargs_hash}')
        for key in file.keys():
           # print(key, file[key])
            #compare = self.h5_group_to_dict(file[key]['kwargs'])
            # add seed to compare to avoid issues TODO fix logic
            if kwargs_hash == file[key]['kwargs_hash'][()].decode():
                print(f'matching kwargs_hash found')
                return file[key]
            #compare['seed'] = kwargs['seed']
            # if kwargs == compare:
            #     return file[key]
        return None

    def h5_group_to_dict(self,
                         h5_group:h5py.Group):
        """Converts a h5py group to a dictionary specific to Sb3_DataClass.py kwargs."""
        dictionary = {}
        for key, item in h5_group.items():
            if isinstance(item, h5py.Dataset):
                if key == "policy" or key == "device":
                    dictionary[key] = item[()].decode() # undo binary for strings

                else:
                    dictionary[key] = item[()]  # Read dataset into a NumPy array
            elif isinstance(item, h5py.Group):
                dictionary[key] = self.h5_group_to_dict(item)  # Recursive call
        return dictionary

    def save(self):
        """
        Open file and save run data
        :return: None
        """

        # check that file exists
        if os.path.exists(self.file):
            file = self._open(self.file, 'r+') # need to read for assertions
            # assert we have the correct attributes for scene and algo
            assert file.attrs["Algorithm"] == self.algorithm
            assert file.attrs["Scene"] == self.scene
            my_group = self._find_group(file,
                                        self.run_data['kwargs'])
            # my_group will be h5py.Group if found, None else
            if my_group is None:
                # hparam unique value for adding new hparam
                num = len(file.keys())
                file.create_group(f'HyperParameters:{num}')
               # print('saving to new kwargs_hash')
                self.save_run(self.run_data, file[f'HyperParameters:{num}'])
            else:
                print('saving to found kwargs_hash')
                self.save_run(self.run_data, my_group) # save to found group
            file.close()

        else:  # initialize new h5 file
            print('Starting new h5')
            with h5py.File(self.file, 'w') as file:
                file.attrs["Algorithm"] = self.algorithm
                file.attrs["Scene"] = self.scene
                # define group for hyper param settings
                file.create_group('HyperParameters:0')
                # call to base class method for storing run dict into group
                self.save_run(self.run_data, file['HyperParameters:0'])
                file.close()

