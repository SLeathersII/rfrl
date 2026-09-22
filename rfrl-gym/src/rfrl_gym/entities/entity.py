import numpy as np

class Entity:
    """Abstract base class defining environmental actors and emitter profiles within the RFRL Gym.
    
    This class serves as the behavioral blueprint and metadata container for all simulated 
    radio frequency (RF) nodes. It models the core temporal logic, active structural states, 
    and parameter validation rules required to interface directly with Virginia Tech National 
    Security Institute's (NSI) **PyWaspgen** framework for baseband In-Phase and 
    Quadrature (I/Q) signal synthesis.

    An `Entity` acts as an independent process or exogenous state-machine that populates the 
    shared electromagnetic environment. It enforces multi-tier structural constraints on spectrum 
    access, temporal duty cycling (finite-state activation windows), and strict physical-layer 
    (PHY) modem configuration mapping.

    Parameters
    ----------
    entity_label : str
        A unique alphanumeric system label assigned to track the entity instance across the 
        global environment logs.
    num_channels : int
        The total cardinal dimension of the discretized baseband environment frequency grid.
    channels : list of int
        The authorized subset of frequency channel indices within which the entity is configured 
        or permitted to allocate transmission energy.
    onoff : list of int
        A three-element state control parameter vector `[x, y, z]` regulating the duty-cycle 
        state machine:
        
        * ``x`` (int, {0, 1}): Initial activation operational state (0 for silent/dormant, 
            1 for active emission).
        * ``y`` (int): Active temporal window length (in environment steps) before transitioning 
            to a dormant phase.
        * ``z`` (int): Dormant temporal window length (in environment steps) before transitioning 
            back to an active phase.
    start : int or None
        The absolute discrete environment timestep indicating when this entity initializes 
        execution and enters the state machine loop. If None, defaults to step 0.
    stop : int or None
        The absolute discrete environment timestep marking when this entity terminates execution 
        and permanently ceases spectral footprint generations. If None, defaults to infinity.
    modem_params : dict
        Physical-layer signal parameter dictionary fed into PyWaspgen for downstream digital 
        waveform synthesis. Required key-value pairs include:
        
        * ``type`` (str): Digital modulation scheme configuration (e.g., 'qam', 'psk', 'fsk').
        * ``order`` (int): Modulation alphabet size or constellation cardinality :math:`M` (e.g., 16, 64).
        * ``filter`` (str): Pulse-shaping filter implementation type (e.g., 'RRC' for Root-Raised Cosine).
        * ``center_frequency`` (list of float): A 2-element list `[low, high]` specifying the normalized 
            baseband center frequency bounds restricted to the Nyquist interval :math:`[-0.5, 0.5]`.
        * ``bandwidth`` (float): Normalized spectral bandwidth occupancy coefficient relative to channel allocation.
        * ``start`` (float): Fractional temporal offset within the step interval :math:`(0, 1)` denoting emission onset.
        * ``duration`` (float): Pulse duty cycle duration expressed as a fraction within the step interval :math:`(0, 1)`.

    Attributes
    ----------
    entity_idx : int
        The globally unique integer index designated by the centralized environment factory for tracking tracking array dimensions.
    info : dict or list
        The localized pointer referencing the current global environment telemetry dictionary structure.
    count : int
        The internal scalar step counter tracking temporal residency within the current active/dormant duty cycle block.
    on_flag : int, {0, 1}
        The current operational state flag indicating whether the entity is mathematically capable of active RF emission.

    Notes
    -----
    Subclasses must override the abstract hook methods ``_validate_self()``, ``_get_action()``, 
    and ``_reset()`` to append specialized behavioral strategies (e.g., frequency hoppers, 
    sweeping jammers, or stationary background users) to this architectural chassis.
    """
    def __init__(self, entity_label: str, num_channels: int, channels: list, onoff: list,
                 start: int, stop: int, modem_params: dict):
        """Initialize the base Entity object and validate structural system inputs."""
        self.entity_label = entity_label
        self.num_channels = num_channels
        self.channels = channels
        self.onoff = onoff   
        self.start = start
        self.stop = stop
        self.modem_params = modem_params

        if self.start == None:
            self.start = 0
        if self.stop == None:
            self.stop = np.inf
        self.info = []  

        self.__validate_entity()

    
    def set_entity_index(self, entity_idx):
        """Assign a unique system matrix identifier to this entity profile.

        Parameters
        ----------
        entity_idx : int
            The unique registration index tracking this entity in the centralized simulator matrix.
        """
        self.entity_idx = entity_idx

    def __validate_entity(self):
        """Execute cross-layer validation on base initialization parameters.

        Verifies array dimensionality, checks channel bounds against environment boundaries, 
        and ensures structural typing integrity for the duty cycle state machine parameters.

        Raises
        ------
        AssertionError
            If configuration parameters violate underlying structural types.
        Exception
            If channel index maps outside the global environment spectral grid or temporal indices 
            are non-integers.
        """
        assert isinstance(self.channels, list), f"Parameter 'channels' of entity '{self.entity_label}' must be of type list."
        for channel in self.channels:
            if channel < 0 or channel >= self.num_channels:
                raise Exception(f"Parameter 'channels' of entity '{self.entity_label}' is not valid for the current gym environment.")
       
        # Validate onoff parameter.
        assert self.onoff[0] in [0,1], f"Parameter 'onoff' of entity '{self.entity_label}' is not {0,1}."
        
        # TODO: refactor the below to robust types and f string    
        if (self.onoff[1] < 0 or str(type(self.onoff[1])) != '<class \'int\'>') or (self.onoff[2] < 0 or str(type(self.onoff[2])) != '<class \'int\'>'):
            raise Exception('Parameter \'onoff\' of entity \'{}\' is not valid.'.format(self.entity_label))
        # Validate start and stop parameters.
        if (str(type(self.start)) != '<class \'int\'>'):
            raise Exception('Parameter \'start\' of entity \'{}\' is not valid.'.format(self.entity_label))
        if (str(type(self.stop)) != '<class \'int\'>') and (self.stop != np.inf):
            raise Exception('Parameter \'stop\' of entity \'{}\' is not valid.'.format(self.entity_label))       

        self._validate_self()
                    
    def _validate_self(self):
        raise Exception('Necessary entity function _validate_self() not implemented for entity type {}'.format(type(self).__name__))
    
    # Get the next action from the entity and check whether signal should turn on/off.
    def get_action(self, info):
        """Process the current simulation epoch step and determine the entity's target channel allocation.

        Evaluates global simulation context against temporal execution bounds (`start` and `stop`). 
        If active, delegates to behavioral heuristics and advances the internal duty-cycle 
        state machine tracking registers.

        Parameters
        ----------
        info : dict
            The live environment diagnostic and tracking ledger dictionary mapping state telemetry.

        Returns
        -------
        int
            The targeted discrete frequency bin allocation index for the current step. 
            Returns -1 if the entity falls outside active global steps or is suppressed by the duty cycle.
        """
        self.info = info
        if (self.info['step_number'] >= self.start) and (self.info['step_number'] <= self.stop):
            action = self._get_action()
            self.__cycle_onoff()
            return action
        else:
            return -1
    
    # Check to make sure the entity has an action function.
    def _get_action(self):
        """Internal behavioral execution hook to determine next channel targets.

        Returns
        -------
        int
            The targeted discrete channel allocation action index.

        Raises
        ------
        NotImplementedError
            If the concrete subclass fails to implement the internal spectrum selection strategy.
        """
        raise NotImplementedError(f"Necessary entity function _get_action() not implemented for entity type {type(self).__name__}")
  
    # Reset the entities parameters
    def reset(self, info):
        """Reset the entity state parameters and cycle counters to baseline initial conditions.

        Parameters
        ----------
        info : dict
            The baseline environment diagnostic tracking dictionary provided at the episode transition boundary.
        """
        self.info = info
        self._reset()
        self.count = 0
        self.on_flag = 0

        if self.onoff[0] == 0 and self.onoff[2] != 0:
            self.on_flag = 0
        elif self.onoff[0] == 1 and self.onoff[1] != 0:
            self.on_flag = 1
    
    # Check to make sure the entity has a reset function.
    def _reset(self):
        """Internal state variable reset hook for subclass-specific parameters.

        Raises
        ------
        NotImplementedError
            If the concrete subclass fails to implement an internal clearing function.
        """
        raise NotImplementedError(f"Necessary entity function _reset() not implemented for entity type {type(self).__name__}")
    
    # Toggle the on/off flag to tell the entity whether or not to do an action.
    def __cycle_onoff(self):
        """Advance the internal finite-state machine regulating active emission duty-cycle intervals.
        
        Increments the state residency counter and toggles the on_flag boolean state mapping 
        upon reaching operational window boundaries defined by onoff[1] (pulse duration)
        and onoff[2] (inter-pulse interval/silent window).
        """
        self.count += 1
        if (self.on_flag == 1 and self.count == self.onoff[1]):
            if self.onoff[2] != 0:
                self.on_flag = 0
            self.count = 0
        elif (self.on_flag == 0 and self.count == self.onoff[2]):
            if self.onoff[1] != 0:
                self.on_flag = 1
            self.count = 0