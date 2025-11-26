from .agile_freq import AgileFreq
from .constant_freq import ConstantFreq
from .fixed_hop_freq import FixedHopFreq
from .simple_jammer import SimpleJammer
from .stochastic_constant_freq import StochasticConstantFreq
from .stochastic_hop_freq import StochasticHopFreq


class EntityGenerator():
    """
    Takes in scenario_metadata from JSON file and creates a list of Entity objects with class methods for generating IQ

    FORMAT: {
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
    {
        "constant_freq_1":
        {
            "type": "ConstantFreq",
            "channels": [0],
            "onoff": [1,1,0],
            "modem_params":
            {
                "type": "qam",
                "order": 16,
                "filter": "RRC",
                "center_frequency": [-0.1,0.1],
                "bandwidth": 0.25,
                "start": 0.25,
                "duration": 0.25
            }
        },
        "constant_freq_2": ....
    """
    def __init__(self, scenario_metadata):
        self.valid_entities = {"ConstantFreq": ConstantFreq,
                               "FixedHopFreq": FixedHopFreq,
                               "StochasticHopFreq": StochasticHopFreq,
                               "StochasticConstantFreq": StochasticConstantFreq,
                               "AgileFreq": AgileFreq,
                               "SimpleJammer": SimpleJammer}
        self.deterministic_entities = {"ConstantFreq": ConstantFreq,
                                       "FixedHopFreq": FixedHopFreq,}
        self.dynamic_entities = {"AgileFreq": AgileFreq,
                                 "SimpleJammer": SimpleJammer,
                                 "StochasticConstantFreq": StochasticConstantFreq,  # TODO -- w/seed=deterministic
                                 "StochasticHopFreq": StochasticHopFreq,   # TODO -- w/seed=deterministic
                                 }
        entity_idx = 0
        self.max_steps = scenario_metadata['environment']['max_steps']
        self.burst_list = []
        self.entity_list = []
        for entity in scenario_metadata['entities']:
            entity_idx += 1
            type = scenario_metadata['entities'][entity]['type']
            if type in self.valid_entities.keys():
                entity_kwargs = {"entity_label": entity, "num_channels": scenario_metadata['environment']['num_channels']}
                for param in scenario_metadata['entities'][entity]:
                    if not param == 'type':
                        entity_kwargs[param] = scenario_metadata['entities'][entity][param]

                self.entity_list.append(self.valid_entities[type](**entity_kwargs))
                self.entity_list[-1].set_entity_index(entity_idx)
            else:
                raise NotImplemented(f'{type} not implemented entities need to be in {self.valid_entities.keys()}')

    def get_entity_list(self):
        return self.entity_list

    def gen_scene(self):
        """
        generates the IQ for the entire scene from the deterministic entities
        Returns: Array of IQ data of length num_samples_per_step*num_steps
        """
        # TODO determine if we want to implement a frame buffer/ threading protocol if max_steps is very large
        for step in range(self.max_steps):
            pass

        pass

    def gen_step(self):
        """
        generate the IQ for a step for the non-deterministic entities (must pass in prior steps info)
        Returns: Array of IQ data of length num_samples_per_step

        """
        pass
