from .entity import Entity


class ConstantFreq(Entity):
    """A static, non-agile entity implementing a deterministic fixed-frequency allocation strategy.

    This entity models a standard, legacy primary user or a continuous bit-rate background emitter 
    with a non-adaptive protocol footprint in a cognitive radio network. Unlike spectrum-agile 
    nodes, it ignores environmental feedback and historical channel state occupancy, continuously 
    occupying its primary designated channel resource whenever its internal duty-cycle state 
    machine permits. 

    From a reinforcement learning perspective, this entity introduces a stationary, predictable 
    interference profile within its active channel index, acting as a deterministic baseline obstacle 
    for the learning agent's policy optimization.
    
    Parameters
    ----------
    entity_label : str or Any
            Unique alphanumeric identifier for the node (e.g., 'agile_transceiver_1').
        num_channels : int
            The total number of the discretized baseband environment frequency bins.
        channels : list of int or np.ndarray
            The specific subset of frequency channel indices within which this entity is 
            authorized or hardware-capable of operating. (Currently not integrated with this Entity!)
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
        AgileFreq : For the adaptive, sense-and-avoid counterpart to this static entity.
        """
    def __init__(self, entity_label, num_channels, channels, onoff=[1,1,0], start=None, stop=None, modem_params=None):
        """Initialize the static fixed-frequency emitting entity."""
        super().__init__(entity_label, num_channels, channels, onoff, start, stop, modem_params)

    def _validate_self(self):
        """Validate structural configurations and system dimensions for the entity.

        Raises
        ------
        ValueError
            If initialization parameters violate baseband channel boundaries or state machine rules.
        """
        pass

    def _get_action(self):
        """Compute the next spectrum allocation action using a static channel mapping.

        Evaluates the local state machine to determine if an active spectral emission window 
        is open. If active, the entity deterministically requests allocation for its primary 
        channel index.

        Returns
        -------
        int
            The discrete channel selection action index corresponding to ``channels[0]``. 
            Returns -1 if the entity's internal duty-cycle state machine dictates an inactive 
            or dormant temporal phase.
        """
        if self.on_flag == 0:
            action = -1
        else:
            action = self.channels[0]
        return action

    def _reset(self):
        """Reset the entity's internal tracking metrics to their baseline initialization parameters.

        Because this profile relies on a static, non-adaptive policy mapping, the reset procedure 
        remains a passive operation with no stochastic state re-sampling required.
        """
        pass
        