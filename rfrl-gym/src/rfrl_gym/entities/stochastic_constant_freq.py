import numpy as np
from .entity import Entity

class StochasticConstantFreq(Entity):
    """A static-frequency emitter implementing a stochastic duty-cycle transmission policy.

    This entity models a intermittent primary user or bursty network node occupying a fixed frequency allocation. 
    While bound to a single structural channel, its transmission profile is governed by a Bernoulli-like process 
    parameterized by a temporal utilization probability (``percent_on``). 

    From an RF perspective, this entity acts as an un-orthogonal, co-channel interference source that switches 
    probabilistically between an active emission state (radiating its configured physical-layer I/Q footprint) 
    and a zero-emission backoff state. 

    From a reinforcement learning perspective, this behavior introduces stationary yet stochastic environment 
    dynamics. It forces the learning agent to generalize over time-varying channel availabilities, treating 
    the targeted frequency bin as a stochastic resource with an expected availability factor of :math:`1 - p`.

    Parameters
    ----------
    entity_label : str
        Unique alphanumeric system label assigned to track the entity instance across the 
        global environment logs.
    num_channels : int
        The total cardinal dimension of the discretized baseband environment frequency grid.
    channels : list of int
        The specific subset of frequency channel indices allocated to this entity. The first element 
        (``channels[0]``) defines the fixed target for static, probabilistic transmission.
    onoff : list of int, default=[1,1,0]
        A three-element state control parameter vector `[x, y, z]` regulating the deterministic duty-cycle 
        state machine framework. Note that the stochastic ``percent_on`` filter overlays this baseline.
        
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
        
        * ``type`` (str): Digital modulation scheme configuration (e.g., 'qam', 'psk').
        * ``order`` (int): Modulation alphabet size or constellation cardinality :math:`M`.
        * ``filter`` (str): Pulse-shaping filter implementation type (e.g., 'RRC').
        * ``center_frequency`` (list of float): A 2-element list `[low, high]` specifying the normalized 
          baseband center frequency bounds restricted to the Nyquist interval :math:`[-0.5, 0.5]`.
        * ``bandwidth`` (float): Normalized spectral bandwidth occupancy coefficient relative to channel allocation.
        * ``start`` (float): Fractional temporal offset within the step interval :math:`(0, 1)` denoting emission onset.
        * ``duration`` (float): Pulse duty cycle duration expressed as a fraction within the step interval :math:`(0, 1)`.
    percent_on : float, default=1.0
        The latent temporal transmission probability :math:`p \\in [0, 1]` defining the statistical 
        fraction of environment epochs during which the entity activates its emission waveform. 
        A value of 1.0 translates to 100% continuous utilization (equivalent to a completely deterministic 
        fixed emitter).

    Attributes
    ----------
    percent_on : float, in [0,1]
        The bounded probability metric tracking the statistical utilization rate of the spectrum channel. 

    See Also
    --------
    ConstantFreq : The non-stochastic, completely deterministic variant of this fixed-frequency emitter.
    """
    def __init__(self, entity_label:str, num_channels: int, channels: list, onoff: list = [1,1,0],
                 start: int = None, stop: int = None, modem_params: dict = None,  percent_on: float = 1.0):
        """Initialize the stochastic fixed-frequency emitting entity."""
        self.percent_on = percent_on
        super().__init__(entity_label, num_channels, channels, onoff, start, stop, modem_params)


    def _validate_self(self):
        """Validate structural configurations and internal parameters specific to the stochastic profile."""
        assert self.percent_on >= 0 and self.percent_on <=1, f"Parameter 'percent_on' of entity '{self.entity_label}' is not valid"
        

    def _get_action(self):
        """Sample a uniform distribution to compute the instantaneous transmission state.

        Evaluates a pseudorandom number draw against the utilization parameter ``percent_on``. 
        If the threshold is exceeded, the entity enters a zero-emission backoff state; otherwise, 
        it yields its fixed allocated channel index.

        Returns
        -------
        int
            The targeted discrete frequency bin allocation index (``channels[0]``) if selected 
            for active transmission. Returns -1 if the random threshold demands a backoff state or if 
            the base duty-cycle state machine forces dormancy.
        """
        rand_num = np.random.uniform(0,1)
        if rand_num > self.percent_on:
            return -1
        else:
            return self.channels[0]

    def _reset(self):
        """Reset internal sequence metrics to baseline parameters at the episode boundary."""
        pass