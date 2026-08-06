import numpy as np

class Detector:
    """Abstract base class for all detector implementations in the RFRL gym.

    This class defines the standard interface for detector objects, which are
    responsible for processing the environment's state information to produce
    sensing results. It is designed to be subclassed, and concrete detector
    implementations must override the `_get_sensing_results` and `_reset`
    methods.

    Parameters
    ----------
    num_channels : int
        The number of frequency channels the detector is configured to monitor.

    Attributes
    ----------
    num_channels : int
        The number of frequency channels.
    info : dict
        A dictionary containing the current state of the RF environment, passed
        during the call to `get_sensing_results`. This typically includes
        information about active transmitters and their properties.

    Notes
    -----
    This class should not be instantiated directly. Instead, create a subclass
    that inherits from `Detector` and implements the required abstract methods.

    See Also
    --------
    EnergyDetector : An example of a concrete implementation of this class.

    """
    def __init__(self, num_channels):
        self.num_channels = num_channels
    
    def get_sensing_results(self, info):
        """Processes environment info and returns sensing results.

        This is the main public method for the detector. It acts as a wrapper
        that receives the environment's state, stores it, and then calls the
        internal `_get_sensing_results` method where the core sensing logic
        is implemented.

        Parameters
        ----------
        info : dict
            A dictionary containing the ground truth state of the environment at
            the current timestep. This may include data on transmitter
            locations, frequencies, power levels, and other relevant metadata.

        Returns
        -------
        np.ndarray
            The sensing results, typically a NumPy array. The shape and data
            type of the array depend on the specific detector implementation
            (e.g., an array of power levels for an energy detector).

        """
        self.info = info
        sensing_results = self._get_sensing_results()
        return sensing_results
    
    # Check to make sure the detector has a sensing function
    def _get_sensing_results(self):
        """Performs the core sensing logic. Must be implemented by subclasses.

        This abstract method is intended to house the specific algorithm for
        the detector (e.g., calculating energy in each channel). It can make
        use of `self.info` which is set by the `get_sensing_results` wrapper.

        Raises
        ------
        NotImplementedError
            If the method is not overridden in a subclass.

        """
        raise Exception('Necessary function _get_sensing_results() not implemented for detector')

    # Check to make sure the detector has a reset function.
    def _reset(self):
        """Resets the internal state of the detector. Must be implemented by subclasses.

        This abstract method should be used to reset any internal state of the
        detector, such as integrators, timers, or historical data buffers. It
        is typically called when the environment is reset.

        Raises
        ------
        NotImplementedError
            If the method is not overridden in a subclass.

        """
        raise Exception('Necessary detector function _reset() not implemented for detector')
