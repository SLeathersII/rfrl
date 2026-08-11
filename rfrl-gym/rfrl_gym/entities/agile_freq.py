import numpy as np
from .entity import Entity


class AgileFreq(Entity):
    """A reactive, spectrum-agile entity implementing sense-and-avoid frequency hopping.

    This entity models a dynamic, non-stationary background user or competitor. 
    It employs an opportunistic spectrum access (OSA) strategy, 
    using oracle action_history from the prior epoch to execute a 
    reactive frequency-hopping pattern. When its active channel undergoes a spectral 
    collision or is occupied by an external emitter, it randomly selects a
    vacant channel from the available empty spectrum pool of the prior step 
    to maintain communication continuity.

    Parameters
    ----------
    entity_label : str or Any
        Unique alphanumeric identifier for the node (e.g., 'agile_transceiver_1').
    num_channels : int
        The total number of the discretized baseband environment frequency bins.
    channels : list of int or np.ndarray
        The specific subset of frequency channel indices within which this entity is 
        authorized or hardware-capable of operating.
    onoff : list of int, default=[1, 1, 0]
        A structural parameter vector `[x, y, z]` regulating the duty-cycle state machine:
        
        * ``x`` (int, {0, 1}): Initial operational state indicator (0 for dormant/silent, 
          1 for active emission).
        * ``y`` (int): Temporal epoch window duration (in simulation steps) before the entity 
          transitions from active to dormant.
        * ``z`` (int): Temporal epoch window duration (in simulation steps) before the entity 
          transitions from dormant back to active.
    start : int, optional
        The absolute discrete environment timestep indicating when this entity initializes 
        execution and enters the state machine loop. If None, defaults to step 0.
    stop : int, optional
        The absolute discrete environment timestep marking when this entity terminates execution 
        and permanently ceases spectral footprint generations.
    modem_params : dict, optional
        Physical-layer configuration payload for deterministic in-phase and quadrature (I/Q) 
        signal generation, containing the following keys:
        
        * ``type`` (str): Digital modulation scheme (e.g., 'qam', 'psk', 'ask').
        * ``order`` (int): number of symbols.
        * ``filter`` (str): Pulse-shaping filter implementation (e.g., 'root_raised_cosine -- RRC').
        * ``center frequency`` (list of float): Normalized baseband center frequency boundary 
          pair restricted to the Nyquist interval :math:`[-0.5, 0.5]`.
        * ``bandwidth`` (float): Normalized spectral occupancy factor relative to channel capacity (1 occupies entire channel, 3 occupies surrounding channels...).
        * ``start`` (float): Fractional temporal offset within the step interval :math:`(0, 1)` 
          denoting emission onset.
        * ``duration`` (float): Pulse duty cycle duration expressed as a fraction within :math:`(0, 1)`.

    Attributes
    ----------
    current_channel : int
        The instantaneous discrete frequency bin index assigned for active baseband transmission.

    See Also
    --------
    RewardMode : For tracking the downstream objective optimization metrics of these transitions.
    """
    def __init__(self, entity_label, num_channels, channels, onoff=[1,1,0], start=None, stop=None, modem_params=None):
        """Initialize the spectrum-agile frequency-hopping entity."""
        super().__init__(entity_label, num_channels, channels, onoff, start, stop, modem_params)
    
    def _validate_self(self):
        """Validate structural configurations and system dimensions for the entity.

        Raises
        ------
        ValueError
            If initialization parameters violate baseband channel boundaries or state machine rules.
        
         Notes
            -----
        Not implemented for this Class -- super().__init__ executes Entity version (polymorphism)
        """
        pass

    def _get_action(self):
        """Compute the next spectrum allocation action using historical oracle sense-and-avoid heuristics.

        Evaluates the local state machine and cross-references historical environment 
        telemetry to determine the optimal frequency channel selection vector for the upcoming 
        temporal snapshot.

        Returns
        -------
        int
            The discrete channel selection action index. Returns -1 if the entity is 
            in an inactive duty-cycle window, a forced backoff state, or if the entire 
            discretized spectrum pool undergoes a complete system blockages (all channels occupied).
        """
        if self.on_flag == 0:
            return -1
        elif self.on_flag == 1:
            occupied_channels = np.zeros(self.num_channels)
            # Map historical agent action from the prior epoch to the channel occupancy vector
            if self.info['action_history'][0][self.info['step_number']-1] != -1:
                occupied_channels[self.info['action_history'][0][self.info['step_number']-1]] = 1
            
            # Determine which channels are occupied not by this entity
            for idx in range(self.num_channels):
                if (self.info['true_history'][self.info['step_number']-1][idx] != self.entity_idx) and (self.info['true_history'][self.info['step_number']-1][idx] != 0):
                    occupied_channels[idx] = self.info['true_history'][self.info['step_number']-1][idx] > 0
           
            # Forced backoff condition under total spectral saturation
            if sum(occupied_channels) == self.num_channels:
                return -1
            # Execute stochastic channel transition upon active-channel interference detection
            elif occupied_channels[self.current_channel] == 1:
                self.current_channel = np.random.choice(np.where(occupied_channels == 0)[0]) # TODO add conditional for self.channels? enforce restricted domain for entity
            return self.current_channel

    def _reset(self):
        """Reset the entity's internal state machine and sample a stochastic initial channel.

        Restores baseline parameters at the episode boundary, evaluating a uniformly 
        distributed random initial condition across the entity's allocated operational spectrum.
        """
        self.current_channel = np.random.choice(self.channels)