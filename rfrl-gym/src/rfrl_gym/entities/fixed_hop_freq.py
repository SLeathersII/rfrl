import numpy as np
from .entity import Entity


class FixedHopFreq(Entity):
    """A deterministic, cyclical frequency-hopping entity with optional pseudo-random pattern permutation.
    
        This entity models a structured electronic counter-countermeasure (ECCM) system or a traditional 
        frequency-hopping spread spectrum (FHSS) transceiver. It iterates through an explicitly defined 
        and bounded subset of baseband channels. Depending on the configuration of the pattern structure, 
        the entity progresses through the assigned channel sequence either lineally or via a frozen, 
        shuffled permutation established at initialization.
    
        From a reinforcement learning perspective, this entity introduces a structured, periodic, 
        and highly non-stationary (yet predictable) interference trajectory into the Markov Decision 
        Process (MDP), challenging the learning agent to map out and adapt to cyclical temporal footprints.
    
        Parameters
        ----------
        entity_label : str
            Unique alphanumeric system label assigned to track the entity instance across the 
            global environment logs. (e.g., 'fixed_hop_freq_1')
        num_channels : int
            The total number of the discretized baseband environment frequency bins.
        channels : list of int
            The discrete subset of authorized frequency channel indices defining the entity's 
            frequency-hopping pattern pool.
        onoff : list of int, default=
            A three-element state control parameter vector `[x, y, z]` regulating the duty-cycle 
            state machine:
            
            * ``x`` (int, {0, 1}): Initial activation operational state (0 for silent/dormant, 
              1 for active emission).
            * ``y`` (int): Active temporal window length (in environment steps) before transitioning 
              to a dormant phase.
            * ``z`` (int): Dormant temporal window length (in environment steps) before transitioning 
              back to an active phase.
        start : int, optional
            The absolute discrete environment timestep indicating when this entity initializes 
            execution and enters the state machine loop. If None, defaults to step 0.
        stop : int, optional
            The absolute discrete environment timestep marking when this entity terminates execution 
            and permanently ceases spectral footprint generations. If None, defaults to infinity.
        modem_params : dict, optional
            Physical-layer signal parameter dictionary fed into PyWaspgen for downstream digital 
            waveform synthesis. Required key-value pairs include:
            
            * ``type`` (str): Digital modulation scheme configuration (e.g., 'qam', 'psk', 'fsk').
            * ``order`` (int): number of symbols.
            * ``filter`` (str): Pulse-shaping filter implementation type (e.g., 'RRC' for Root-Raised Cosine).
            * ``center_frequency`` (list of float): Normalized baseband center frequency boundary 
              pair restricted to the Nyquist interval :math:`[-0.5, 0.5]`.
            * ``bandwidth`` (float): Normalized spectral bandwidth occupancy coefficient relative to channel capacity (1 occupies entire channel, 3 occupies surrounding channels...).
            * ``start`` (float): Fractional temporal offset within the step interval :math:`(0, 1)` denoting emission onset.
            * ``duration`` (float): Pulse duty cycle duration expressed as a fraction within the step interval :math:`(0, 1)`.
            * ``rand_hop`` (int): A binary flag In {0, 1} controlling the hopping sequence permutation logic; 1 randomizes the order of `channels` list
    
        Attributes
        ----------
        rand_hop : int
            The configuration flag indicating whether channel permutation scrambling is enabled.
    
        See Also
        --------
        AgileFreq : For a reactive, sense-and-avoid counterpart that dynamically shifts channels based on collisions.
        """
    def __init__(self, entity_label:str, num_channels: int, channels: list, onoff: list = [1,1,0],
                 start: int = None, stop: int = None, modem_params: dict = None, rand_hop: int = 1):
        """Initialize the fixed-pattern frequency-hopping transceiver entity."""
        
        self.rand_hop = rand_hop
        super().__init__(entity_label, num_channels, channels, onoff, start, stop, modem_params)

        #self.rng = np.random.default_rng()
        if self.rand_hop == 1:
            np.random.shuffle(self.channels)
            #self.rng.shuffle(self.channels)
    
    def _validate_self(self):
        """Validate structural configurations specific to the fixed-hop profile.

        Verifies that the frequency-hopping trajectory toggle is bound to binary logic constraints 
        to ensure proper track unrolling at initialization.

        Raises
        ------
        AssertionError
            If the ``rand_hop`` evaluation attribute is configured outside the binary set ``{0, 1}``.
        """
        assert self.rand_hop in [0,1], f"Parameter 'rand_hop' of entity '{self.entity_label}' is not valid"

    def _get_action(self):
        """Compute the next spectrum allocation action using a cyclical sequence stepping index.

        Evaluates the active temporal emission window flag. If active, it systematically advances 
        the modular channel index at the initialization boundary of each duty-cycle pulse frame, 
        progressing predictably through the assigned frequency constellation array.

        Returns
        -------
        int
            The discrete channel selection action index corresponding to the current state 
            of the hopping trajectory. Returns -1 if the entity's internal duty-cycle 
            state machine dictates a passive or dormant interval.
        """
        if self.on_flag == 0:
            return -1
        elif self.on_flag == 1:
            if self.count == 0:
                self.channel_index += 1
            if self.channel_index == len(self.channels):
                self.channel_index = 0
            return self.channels[self.channel_index]

    def _reset(self):
        """Reset the internal sequence pointer to its baseline initialization boundary.

        Restores the hopping pattern tracking index to its pre-emission boundary state. This ensures 
        that the subsequent global environment transition steps cleanly into the first element of 
        the allocated spectrum array.
        """
        self.channel_index = -1