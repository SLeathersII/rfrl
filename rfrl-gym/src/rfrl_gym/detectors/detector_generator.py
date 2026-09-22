import numpy as np


from .detector import Detector
from .energy_detector import EnergyDetector


class DetectorGenerator():
    """Factory class for creating energy detector objects for the RFRL gym.

    This class parses scenario metadata to instantiate and configure the
    appropriate energy detector. The created detector is intended to be used
as an observation wrapper in a Gymnasium environment, providing IQ data or a
    label map for a sensing algorithm.

    Parameters
    ----------
    scenario_metadata : dict
        A dictionary, typically loaded from a JSON file, containing the
        configuration for the simulation environment. The structure should
        include an 'environment' key with detector specifications.

    Attributes
    ----------
    valid_detectors : dict
        A dictionary mapping valid detector type strings to their respective
        class implementations (e.g., {'EnergyDetector': EnergyDetector}).
    detector_type : str
        The type of detector to be instantiated, extracted from the
        scenario metadata.
    detector_kwargs : dict
        A dictionary of keyword arguments used to initialize the specified
        detector.
    detector : object
        The instantiated detector object, ready to be used by the
        environment.

    Raises
    ------
    NotImplementedError
        If the `detector_type` specified in the scenario metadata is not
        found in the `valid_detectors` dictionary.

    Examples
    --------
    >>> # Assuming EnergyDetector class is defined
    >>> class EnergyDetector:
    ...     def __init__(self, num_channels):
    ...         self.num_channels = num_channels
    ...     def __repr__(self):
    ...         return f"EnergyDetector(num_channels={self.num_channels})"
    ...
    >>> scenario_data = {
    ...     "environment": {
    ...         "num_channels": 10,
    ...         "max_steps": 30,
    ...         "observation_mode": "detect",
    ...         "reward_mode": "dsa",
    ...         "target_entity": "fixed_hop_freq_1",
    ...         "detector": {
    ...             "energy_detector": {
    ...                 "type": "EnergyDetector",
    ...                 "kwargs": {
    ...                     "num_channels": 10
    ...                 }
    ...             }
    ...         }
    ...     },
    ...     "entities": {}
    ... }
    >>> generator = DetectorGenerator(scenario_metadata=scenario_data)
    >>> detector_instance = generator.detector_out()
    >>> print(detector_instance)
    EnergyDetector(num_channels=10)

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






