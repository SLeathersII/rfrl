import numpy as np
from sphinx.cmd.quickstart import valid_dir

from .detector import Detector
from .energy_detector import EnergyDetector


class DetectorGenerator():
    """
    Takes in the scenario_metadata from the JSON scenario for RFRL gym and outputs the sensor to be used by the gym
    FORMAT:
    {
    "environment":
    {
	    "num_channels": 10,
	    "max_steps": 30,
	    "observation_mode": "detect",
	    "reward_mode": "dsa",
        "target_entity": "fixed_hop_freq_1",
        "detector":
        {
            "energy_detector":
            {
                "type": "EnergyDetector",
                "kwargs":
                {
                    "num_channels": 10
                }
            }
        }
    },
    "entities":
    """
    def __init__(self, scenario_metadata):
        # need to import all detectors and keep in immutable object
        self.valid_detectors = {'EnergyDetector': EnergyDetector}

        self.detector_type = scenario_metadata['environment']['detector']['energy_detector']['type']
        self.detector_kwargs = scenario_metadata['environment']['detector']['energy_detector']['kwargs']
        if self.detector_type in self.valid_detectors.keys():
            self.detector = self.valid_detectors[self.detector_type](**self.detector_kwargs)
        else:
           # self.detector = EnergyDetector(num_channels=scenario_metadata['environment']['num_channels'])
            raise NotImplementedError(f'{self.detector_type} not implemented must be in {self.valid_detectors.keys()}')



    def detector_out(self):
        return self.detector






