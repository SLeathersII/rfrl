import numpy as np
from .entity import Entity


class SimpleJammer(Entity):
    """A reactive electronic warfare (EW) countermeasure entity implementing dynamic spot jamming.

    This entity models a simple jamming node that continuously monitors the true channel occupancy
    of the prior step and uniformly samples from the occupied channels to naively jam. 
    (there is no prediction or targeting logic)

    The asset features an advanced repetition-avoidance mechanism designed to emulate non-stationary, 
    unpredictable countermeasure behavior, preventing the learning agent from exploiting trivial 
    stationary convergence states.

    Parameters
    ----------
    entity_label : str
        Unique alphanumeric system label assigned to track the entity instance across the 
        global environment logs.
    num_channels : int
        The total cardinal dimension of the discretized baseband environment frequency grid.
    channels : list of int
        The authorized subset of frequency channel indices within which this jammer is 
        configured or hardware-capable of radiating energy.
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
        
        * ``type`` (str): Digital modulation scheme configuration (e.g., 'noise', 'fsk').
        * ``order`` (int): Modulation alphabet size or constellation cardinality :math:`M`.
        * ``filter`` (str): Pulse-shaping filter implementation type (e.g., 'RRC').
        * ``center_frequency`` (list of float): A 2-element list `[low, high]` specifying the normalized 
          baseband center frequency bounds restricted to the Nyquist interval :math:`[-0.5, 0.5]`.
        * ``bandwidth`` (float): Normalized spectral bandwidth occupancy coefficient relative to channel allocation.
        * ``start`` (float): Fractional temporal offset within the step interval :math:`(0, 1)` denoting emission onset.
        * ``duration`` (float): Pulse duty cycle duration expressed as a fraction within the step interval :math:`(0, 1)`.
    avoid_repeats : bool, default=True
        A constraint toggle that penalizes continuous dwell time on a single channel. If True, 
        forces the target selection heuristic to pivot to alternative occupied frequency bins 
        over subsequent simulation epochs if multiple collision opportunities exist.

    Attributes
    ----------
    avoid_repeats : bool
        The structural parameter governing continuous frequency dwell constraints.
    current_channel : int or None
        The discrete channel index currently or most recently subjected to electronic attack.

    See Also
    --------
    AgileFreq : An adaptive communications entity that uses sense-and-avoid heuristics to escape this jammer.
    """
    def __init__(self, entity_label:str, num_channels: int, channels: list, onoff: list = [1,1,0],
                 start: int = None, stop: int = None, modem_params: dict = None,  avoid_repeats: bool =True):
        """Initialize the electronic countermeasure spot-jamming entity."""
        self.avoid_repeats = avoid_repeats
        super().__init__(entity_label, num_channels, channels, onoff, start, stop, modem_params)
    
    def _validate_self(self):
        """Validate structural configurations and internal parameter sets for the jammer."""
        pass       
    
    def _get_action(self):
        """Compute the next spectrum attack action using historical sense-and-disrupt metrics.

        Evaluates the environment true_history and samples a prior occupied channel to transmit jamming energy.  

        Returns
        -------
        int
            The targeted discrete frequency bin allocation index chosen for jamming injection. 
            Returns -1 if the jammer is in a passive duty-cycle frame.
        """
        if self.on_flag == 0 or self.info['step_number'] == 0:
            return -1
        elif self.on_flag == 1:  
            occupied_channels = np.zeros(self.num_channels)
            if self.info['action_history'][0][self.info['step_number']-1] != -1:
                occupied_channels[self.info['action_history'][0][self.info['step_number']-1]] = 1
            for idx in range(self.num_channels):
                if (self.info['true_history'][self.info['step_number']-1][idx] != self.entity_idx) and (self.info['true_history'][self.info['step_number']-1][idx] != 0):
                    if occupied_channels[idx] == 1 or self.info['true_history'][self.info['step_number']-1][idx] == self.info["num_entities"]+1:
                        if self.current_channel == idx:
                            occupied_channels[idx] = 2
                        else:
                            occupied_channels[idx] = 0
                    else:
                        occupied_channels[idx] = self.info['true_history'][self.info['step_number']-1][idx] > 0
            
            if sum(occupied_channels) == 0:
                return -1
            else:
                if (self.avoid_repeats == True) and (len(np.where(occupied_channels == 1)[0]) > 0):
                    self.current_channel = np.random.choice(np.where(occupied_channels == 1)[0])
                else:
                    self.current_channel = np.random.choice(np.where(occupied_channels > 0)[0])

                return self.current_channel

    def _reset(self):
       """Reset the internal tracking counter and historical targeted channel vector.

        Resets the current channel for the entity during reset environment call.
        """ 
       self.current_channel = None