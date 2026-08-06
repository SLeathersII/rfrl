import numpy as np
from .detector import Detector

class EnergyDetector(Detector):
    """A concrete detector that identifies signal presence based on energy levels.

    This class implements the `Detector` interface to provide a simple energy-
    based sensing mechanism. It operates by comparing pre-calculated energy
    values for each channel against a fixed threshold to produce a binary
    (presence/absence) result.

    This implementation assumes that the actual energy calculation is performed
    elsewhere in the simulation (e.g., a renderer) and made available within
    the `info` dictionary.

    Parameters
    ----------
    num_channels : int
        The number of frequency channels the detector is configured to monitor.

    Attributes
    ----------
    num_channels : int
        The number of frequency channels.
    info : dict, optional
        A dictionary containing the current state of the RF environment,
        populated during the call to `get_sensing_results`.

    See Also
    --------
    Detector : The abstract base class that this class implements.

    """
    def __init__(self, num_channels):
        super().__init__(num_channels)

    def _validate_self(self):
        pass

    # Note: when we get sensing results, they are the results for the frame rendered
    # immediately before the environment's step() method was called
    # Since the previous frame's sensing results were calculated in pyqt_renderer.py,
    # we can just fetch it from self.info['sensing_energy_history']
    def _get_sensing_results(self):
        """Generates binary sensing results by thresholding energy values.

        This method implements the core logic for the energy detector. It
        retrieves a vector of pre-calculated energy values for the current
        simulation step from `self.info['sensing_energy_history']`. It then
        iterates through each channel, applies a hard-coded threshold to
        determine signal presence (1) or absence (0), and records this binary
        result in `self.info['sensing_history']`.

        Notes
        -----
        The sensing results correspond to the state of the environment in the
        frame rendered *before* the agent's `step()` action is executed. The
        energy values are fetched directly from the `info` dictionary, not
        calculated within this method.

        Returns
        -------
        dict
            The modified `info` dictionary, now updated with the binary
            sensing results for the current step under the
            `sensing_history` key.

        """
        for k in range(self.num_channels):
            sensed_result = self.info['sensing_energy_history'][self.info['step_number']][k]
            if sensed_result > 0.001:
                self.info['sensing_history'][self.info['step_number']][k] = 1
            else:
                self.info['sensing_history'][self.info['step_number']][k] = 0
        return self.info
    
    def _reset(self):
        pass

