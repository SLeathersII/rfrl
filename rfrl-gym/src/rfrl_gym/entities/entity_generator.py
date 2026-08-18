from .agile_freq import AgileFreq
from .constant_freq import ConstantFreq
from .fixed_hop_freq import FixedHopFreq
from .simple_jammer import SimpleJammer
from .stochastic_constant_freq import StochasticConstantFreq
from .stochastic_hop_freq import StochasticHopFreq


class EntityGenerator():
    """An instantiation factory for entities in RFRL-gym.

    This class parses JSON-serialized scenario metadata to dynamically orchestrate, 
    instantiate, and catalog the RF entities populating the Gymnasium environment. 
    It maintains structural registries categorizing entities by their mathematical 
    determinism, distinguishing between stationary/apriori-mappable emitters and reactive, 
    non-stationary agents (e.g., jammers and adaptive spectrum nodes).

    This is designed to work with PyWaspGen and simulate IQ data on a complex environment
    determined by the entities generated. Currently looking at refactoring towards the IQ simulator backbone API
    such that other IQ generators could use the same framework and integrate with RFRL gym. 

    Parameters
    ----------
    scenario_metadata : dict
        A nested dictionary structure adhering to the environment configuration schema. 
        Must include an ``'environment'`` key specifying channelization geometry and temporal 
        horizons, alongside an ``'entities'`` key mapping unique alphanumeric labels to 
        their respective physical-layer properties, duty-cycle parameters, and modulation parameters.

    Attributes
    ----------
    valid_entities : dict[str, Type[Entity]]
        A global lookup registry mapping structural string type identifiers to their concrete 
        ``Entity`` subclass definitions.
    deterministic_entities : dict[str, Type[Entity]]
        A subset registry identifying entities whose transmission vectors depend exclusively 
        on time or static internal counters, enabling apriori trajectory unrolling.
    dynamic_entities : dict[str, Type[Entity]]
        A subset registry identifying stochastic, reactive, or interactive entities whose 
        frequency-allocation trajectories cannot be pre-computed.
    max_steps : int
        The temporal episode horizon specifying the terminal state boundaries of the underlying MDP.
    entity_list : list of Entity
        An ordered array containing the fully initialized, active entity object instances 
        populating the simulation.
    burst_list : list
        An internal collection buffer intended for tracking temporal burst structures and 
        pulse-repetition intervals (PRIs).

    Examples
    --------
    >>> schema = {
    ...     "environment": {"num_channels": 10, "max_steps": 30},
    ...     "entities": {
    ...         "constant_freq_1": {
    ...             "type": "ConstantFreq", "channels":, "onoff":,
    ...             "modem_params": {"type": "qam", "order": 16, "filter": "RRC", 
    ...                              "center_frequency": [-0.1,0.1], "bandwidth": 0.25, 
    ...                              "start": 0.25, "duration": 0.25}
    ...         }
    ...     }
    ... }
    >>> generator = EntityGenerator(schema)
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
                raise NotImplementedError(
                    f'{type} not implemented. Entities must be members of {list(self.valid_entities.keys())}'
                )

    def get_entity_list(self):
        """Retrieve the flat collection of all instantiated RF simulation objects.

        Returns
        -------
        list of Entity
            The structured collection of active entity subclasses participating in the environment.
        """
        return self.entity_list


    # TODO -- implement methods to support deterministic scene generation (awaiting memory/compute optimizations) better for many runs of same scene
    def gen_scene(self):
        """Synthesize the complete continuous baseband I/Q vector for all deterministic entities for the full episode.

        This method unrolls the state trajectories of all registered deterministic background 
        emitters across the entire temporal duration of the episode horizon. By pre-computing 
        and vectorizing the linear superposition of these non-reactive signals, the environment 
        optimizes performance, rendering a baseline electromagnetic framework that remains constant 
        across separate training iterations of the same scenario seed. SEE: gymnasium.vector.utils.shared_memory

        Returns
        -------
        np.ndarray
            A 1D complex-valued NumPy array containing the aggregated, time-domain digital 
            baseband I/Q sample sequence spanning the full scene duration.
            Shape matches ``(max_steps * num_samples_per_step,)``.

        Notes
        -----
        This block-generation path completely bypasses reactive interactions, acting strictly 
        as an optimization mechanism for invariant channel conditions. Intended to use with gymnasium.vector.utils.shared_memory
        to reduce CPU load when vectorizing many runs of the same environment reducing redundant computation. 
        """
        for step in range(self.max_steps):
            pass
        pass

    def gen_step(self):
        """Synthesize the instantaneous complex baseband I/Q matrix for a single execution step.

        This method sequentially computes the discrete time-domain waveforms generated by 
        the dynamic, stochastic, and reactive entities for the current environment step. Because 
        these entities (such as sense-and-avoid frequency hoppers or active tracking jammers) 
        evaluate state transition decisions based on closed-loop feedback from the preceding step, 
        their physical signals must be synthesized interactively alongside the RL agent's chosen action.

        Returns
        -------
        np.ndarray
            A 1D complex-valued NumPy array representing the mathematical superposition 
            of active dynamic emissions and interference waveforms generated during this 
            isolated temporal step. Shape matches ``(num_samples_per_step,)``.
        """
        pass
