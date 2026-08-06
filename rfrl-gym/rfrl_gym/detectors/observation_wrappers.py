from ..envs import RFRLGymIQEnv2
from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar, Union
from gymnasium.spaces import Box, Discrete, Space, MultiBinary
from gymnasium import Env, Wrapper
from .detector import Detector
import numpy as np
from scipy.ndimage import convolve1d
import scipy.signal as signal

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")
WrapperObsType = TypeVar("WrapperObsType")




class Sensor(Wrapper[WrapperObsType, ActType, ObsType, ActType]):
    """An abstract observation wrapper for simulating RF sensors in Gymnasium.

    This class serves as a base for creating sensor models within the RFRL
    gym framework. It intercepts the raw state information from the underlying
    RF environment and transforms it into a processed observation, simulating
    the output of a physical sensor.

    Concrete sensor implementations (e.g., EnergyDetector, Cyclostationary
    Detector) should inherit from this class and must implement the
    `observation` method to define the specific signal processing logic. The
    resulting observation format must match the `observation_space` defined
    during initialization.

    Parameters
    ----------
    env : Env[ObsType, ActType]
        The Gymnasium environment that simulates the RF spectrum.
    space : Union[Space, str]
        The Gymnasium `Space` defining the structure of the processed
        observation returned by the sensor. If a string is provided, a default
        space may be inferred.

    Attributes
    ----------
    observation_space : Space
        The observation space of the wrapped environment, specifying the format
        of the data passed to the learning agent.

    """
    
    def __init__(self, env: Env[ObsType, ActType],
                 space: Union[Space, str],
                 ):
        Wrapper.__init__(self, env)
        if isinstance(space, Space):
            self.observation_space = space
        # elif space in ['detect', 'classify']:
        #     if space == 'detect':
        #         self.observation_space = Discrete(2**env.unwrapped.num_channels)  # binary for bins
        #     elif space == 'classify':
        #         self.observation_space = Discrete((1+env.unwrapped.num_entities)**env.unwrapped.num_channels)
        else:
            self.observation_space = MultiBinary(self.env.unwrapped.num_channels)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Resets the environment and applies the sensor observation logic.

        This method calls the underlying environment's `reset` method and then
        processes the initial raw observation through the `self.observation`
        method.

        Parameters
        ----------
        seed : int, optional
            The seed to use for the environment's random number generator.
        options : dict, optional
            Additional options to pass to the environment's `reset` method.

        Returns
        -------
        WrapperObsType
            The initial processed observation from the sensor.
        dict
            The auxiliary information dictionary returned by the environment.

        """
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(
        self, action: ActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Executes one step and processes the resulting observation.

        This method passes the agent's action to the underlying environment,
        then transforms the resulting raw signal data using the `self.observation`
        method before returning it to the agent.

        Parameters
        ----------
        action : ActType
            The action taken by the agent.

        Returns
        -------
        WrapperObsType
            The processed observation from the sensor for the current step.
        SupportsFloat
            The reward returned by the environment.
        bool
            A boolean indicating if the episode has terminated.
        bool
            A boolean indicating if the episode has been truncated.
        dict
            The auxiliary information dictionary returned by the environment.

        """
        signal_data, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(signal_data), reward, terminated, truncated, info

    def observation(self, observation: ObsType) -> WrapperObsType:
        """Processes raw RF environment data into a sensor observation.

        This abstract method must be implemented by all subclasses. Its role
        is to define the core signal processing logic of the sensor,
        transforming the raw data from the environment into a structured
        observation for the agent.

        Parameters
        ----------
        observation : ObsType
            The raw observation from the underlying environment, typically
            representing the state of the RF spectrum (e.g., IQ samples,
            power levels, or ground truth metadata).

        Returns
        -------
        WrapperObsType
            The processed observation, which must conform to the wrapper's
            `observation_space`.

        Raises
        ------
        NotImplementedError
            If the method is not overridden in a subclass.

        """
        raise NotImplementedError("Each Sensor must implement its own observation method.")

    def _observation_space_encoder(self, observation_vect):
        """Encodes a processed observation vector for the policy.

        This method is intended for any final encoding step required to format
        the sensor's output vector into the specific shape or type expected
        by the policy network (e.g., flattening a 2D array or converting a
        binary vector to a discrete index).

        Parameters
        ----------
        observation_vect : np.ndarray
            The processed observation vector, typically the output from the
            `observation` method.

        Returns
        -------
        Any
            The finally encoded observation to be passed to the agent's policy.

        Raises
        ------
        NotImplementedError
            If the method is not overridden in a subclass.

        """
        raise NotImplementedError("This sensor does not have an observation space encoder.")

class EnergyDetector(Sensor):
    """ A Gymnasium wrapper that simulates a channelized energy detector.

    This class implements a classical non-coherent energy detector for spectrum
    sensing. It processes wideband I/Q data from the base environment by
    channelizing the spectrum, calculating the signal energy within each
    channel, and applying a threshold to determine channel occupancy.

    The signal processing chain for each channel consists of:
    1. Frequency shifting to baseband the channel of interest.
    2. Low-pass filtering to isolate the channel bandwidth.
    3. Downsampling to the channel's Nyquist rate.
    4. Calculating the average energy of the resulting signal.
    5. Comparing the energy against a detection threshold.

    Parameters
    ----------
    env : Env[ObsType, ActType]
        The Gymnasium environment that provides the raw RF signal data.
    space : {"detect", "classify"}
        A string indicating the encoding mode, used to define the base for
        the integer-based observation encoding in `_observation_space_encoder`.
    threshold : float, optional
        The energy threshold for making a detection decision. A signal with
        normalized energy above this value is considered a detection.
        Default is 0.8.
    sos_filter : np.ndarray, optional
        The coefficients of a digital filter in second-order sections (SOS)
        format. This filter is used for channel selection after basebanding.
        If `None`, a default Butterworth low-pass filter is designed.

    Attributes
    ----------
    threshold : float
        The energy detection threshold.
    sos_filter : np.ndarray
        The SOS filter coefficients used for channelization.
    observation_base : int
        The numerical base used for encoding the observation vector into a
        single integer.

    """

    def __init__(self, env: Env[ObsType, ActType],
                 space: Union[Space, str],
                 threshold: float = 0.8,
                 sos_filter: np.ndarray = None
                 ):
        
        super().__init__(env, space)
        # observation base for encoding method
        if space == 'detect':
            self.observation_base = 2
        elif space == 'classify':
            self.observation_base = 1 + env.unwrapped.num_entities

        self.threshold = threshold
        if sos_filter is None:
            sos_filter = signal.butter(30, 1/self.env.unwrapped.num_channels, output='sos')
        self.sos_filter = sos_filter
        # this introduces a bug it does not pass a pointer
        #self.info = self.env.unwrapped.info

    def _observation_space_encoder(self, observation_vect):
        """Encodes a binary observation vector into a single integer.

        This utility converts the multi-binary channel occupancy vector into a
        unique integer. This can be used to map the observation to a discrete
        Gymnasium space for the policy.

        Parameters
        ----------
        observation_vect : List[int]
            The vector of binary detection results (1 for occupied, 0 for vacant).

        Returns
        -------
        int
            The integer representation of the observation vector.
        """
        observation_int = 0
        for idx in range(len(observation_vect)):
            observation_int += (self.observation_base ** idx) * observation_vect[idx]
        return observation_int

    def observation(self, signal_data: ObsType) -> WrapperObsType:
        """Processes raw I/Q data to produce a channel occupancy vector.

        This method executes the core energy detection algorithm. It takes the
        raw complex signal data from the environment and returns a binary list
        indicating which channels are occupied.

        Parameters
        ----------
        signal_data : np.ndarray
            The raw observation from the environment, expected to be an array
            of shape (2, N), where N is the number of samples and the first
            row is the In-phase (I) component and the second row is the
            Quadrature (Q) component.

        Returns
        -------
        List[int]
            A list of binary values representing the occupancy of each channel.
            `1` indicates the channel energy is above the threshold (occupied),
            and `0` indicates it is below (vacant).
        """
        phase_shift_base = -1j*2*np.pi*self.env.unwrapped.t
        # convert Box signal data into a complex array
        signal_data_step = np.empty(signal_data.shape[1], dtype=np.complex128)
        signal_data_step.real = signal_data[0]
        signal_data_step.imag = signal_data[1]
        sensing_history = []
        for k in range(self.env.unwrapped.num_channels):
            data = signal_data_step * np.exp(phase_shift_base*self.env.unwrapped.fc[k])
            filtered = signal.sosfilt(self.sos_filter, data)
            downsampled = filtered[::self.env.unwrapped.num_channels]
            sensed_result = np.sum(np.abs(downsampled)**2.0)\
                                 /(self.env.unwrapped.samples_per_step/self.env.unwrapped.num_channels)
            self.env.unwrapped.info['sensing_energy_history'][self.env.unwrapped.info['step_number']][k] = sensed_result
            # turn the sensed_result into binary based on threshold
            sensing_history.append((sensed_result > self.threshold)*1)
        self.env.unwrapped.info['observation_history'][self.env.unwrapped.info['step_number']] = sensing_history
        #return self._observation_space_encoder(sensing_history)
        return sensing_history

class OracleMap(Sensor):
    """A perfect, all-knowing 'oracle' sensor for ground truth observations.

    This wrapper acts as an ideal sensor that provides perfect and instantaneous
    knowledge of the true channel occupancy state from the environment. It
    bypasses any signal processing and directly queries the environment's ground
    truth.

    This sensor is useful for theoretical analysis, debugging, and establishing
    an upper performance bound for a reinforcement learning agent. The abstract 
    environment is the ideal use case to avoid simulating IQ.

    Parameters
    ----------
    env : Env[ObsType, ActType]
        The Gymnasium environment that provides the ground truth state.
    space : {"detect", "classify"}
        A string indicating the encoding mode. While used for consistency, it
        does not alter the observation logic of this sensor.

    """
    def __init__(self, env: Env[ObsType, ActType],
                 space: Union[Space, str],
                 ):
        """
        Oracle detection based environment ground truth
        Args:
            env: base environment to wrap
            space: observation space to be encoded to
        """
        super().__init__(env, space)
        # observation base for encoding method
        if space == 'detect':
            self.observation_base = 2
        elif space == 'classify':
            self.observation_base = 1 + env.unwrapped.num_entities
        self.observation_space = MultiBinary(env.unwrapped.num_channels)
    def _observation_space_encoder(self, observation_vect):
        """Returns the true channel occupancy from the environment's state.

        In this implementation, the encoder directly provides the ground truth
        from the environment, bypassing any encoding of the input vector.

        Parameters
        ----------
        observation_vect : np.ndarray
            The observation vector, which is ignored in this implementation.

        Returns
        -------
        np.ndarray
            The true channel occupancy vector from the environment.

        """
        return self.env.unwrapped._get_true_step_occupancy()

    def observation(self, signal_data: ObsType) -> WrapperObsType:
        """Returns the ground truth channel occupancy.

        This method ignores the `signal_data` input and directly retrieves the
        true channel occupancy state from the underlying environment.

        Parameters
        ----------
        signal_data : Any
            The raw observation from the environment, which is ignored.

        Returns
        -------
        np.ndarray
            A NumPy array of binary values representing the true occupancy of
            each channel (ground truth). `1` means occupied, `0` means vacant.

        """
        true_history_step = self.env.unwrapped._get_true_step_occupancy()
        self.env.unwrapped.info['observation_history'][self.env.unwrapped.info['step_number']] = true_history_step
        return true_history_step # self._observation_space_encoder(true_history_step)


class CA_CFAR(Sensor):
    """A Gymnasium wrapper simulating a Cell-Averaging Constant False Alarm Rate (CA-CFAR) detector.

    This sensor processes wideband I/Q data by first channelizing the
    spectrum and then applying a CA-CFAR algorithm within each channel to
    adaptively detect signal presence. CFAR is a common radar and communications
    technique that maintains a constant probability of false alarm by dynamically
    adjusting the detection threshold based on the local noise floor estimate.

    The processing pipeline is as follows:
    1.  **Channelization**: The input wideband signal is demultiplexed into
        individual, narrower channels.
    2.  **PSD Estimation**: For each channel, the Power Spectral Density (PSD)
        is estimated using an FFT.
    3.  **Noise Estimation**: A 1D convolution (moving average) is applied to
        the PSD to estimate the local noise power for each frequency bin.
        Guard cells are used to prevent target energy from biasing the estimate.
    4.  **Threshold Calculation**: An adaptive threshold is calculated by scaling
        the noise estimate with a factor `alpha`, derived from the desired `p_fa`.
    5.  **Detection**: A detection is declared in a frequency bin if its power
        exceeds the adaptive threshold.
    6.  **Occupancy Decision**: A channel is declared occupied if the number of
        bin-level detections exceeds a final decision threshold.

    Parameters
    ----------
    env : Env
        The base Gymnasium environment to be wrapped.
    space : {"detect", "classify"}
        The encoding mode, used to define the base for observation encoding.
    p_fa : float
        The desired Probability of False Alarm. This parameter controls the
        detector's sensitivity and is used to derive the threshold scaling
        factor, `alpha`.
    num_guard_cells : int, optional
        The number of guard cells to use on *each side* of the Cell Under Test
        (CUT). These cells are excluded from the noise estimate to prevent
        target energy leakage. Defaults to 3.
    num_average_cells : int, optional
        The number of training cells to use on *each side* of the CUT for
        estimating the noise floor. The total number of cells in the averaging
        window is `2 * num_average_cells`. Defaults to 12.
    filter : np.ndarray, optional
        The FIR filter coefficients for the anti-aliasing filter used during
        channelization. If `None`, a default Hamming windowed FIR filter is
        designed.

    Attributes
    ----------
    cfar_alpha : float
        The threshold scaling factor, calculated from `p_fa` and the number of
        averaging cells.
    cfar_detection_threshold : float
        A secondary threshold on the number of bin-level detections required
        to declare a channel occupied.
    filter : np.ndarray
        The FIR filter coefficients.
    num_guard_cells : int
        The number of guard cells per side.
    num_average_cells : int
        The number of averaging cells per side.

    """

    def __init__(self, env: Env[ObsType, ActType],
                 space: Union[Space, str],
                 p_fa: float,
                 num_guard_cells: int = 3,
                 num_average_cells: int = 12,
                 filter: np.ndarray = None):
        
        super().__init__(env, space)
        # observation base for encoding method
        if space == 'detect':
            self.observation_base = 2
        elif space == 'classify':
            self.observation_base = 1 + env.unwrapped.num_entities
        self.observation_space = MultiBinary(env.unwrapped.num_channels)
        # compute alpha
        self.cfar_alpha = 2*num_average_cells*(p_fa**(-1/(2*num_average_cells))-1)
        self.cfar_detection_threshold = p_fa * self.env.unwrapped.samples_per_step ## expected number of false alarms
        if filter is None:
            filter = signal.firwin(env.unwrapped.num_channels * 8, 1.0/env.unwrapped.num_channels, window='hamming')
        self.filter = filter

        # store kernel data (numbers are halves which get mirrored to create kernel)
        self.num_guard_cells = num_guard_cells
        self.num_average_cells = num_average_cells

    @staticmethod
    def gen_cfar_conv(num_guard_cells, num_average_cells):
        """Generates the 1D CFAR convolution kernel.

        This kernel is used to perform the sliding window average for noise
        estimation. It contains ones for the averaging cells and zeros for the
        guard cells and the Cell Under Test (CUT).

        Parameters
        ----------
        num_guard_cells : int
            The number of guard cells per side.
        num_average_cells : int
            The number of averaging cells per side.

        Returns
        -------
        np.ndarray
            The symmetric 1D convolution kernel.
        """
        half_conv = np.zeros(num_guard_cells + num_average_cells + 1)
        for avg in range(num_average_cells):
            half_conv[avg] = 1
        return np.concat((half_conv, half_conv[:-1][::-1]))

    def cfar_noise(self, power, num_guard_cells, num_average_cells):
        """Estimates the local noise floor using a sliding window average.

        Parameters
        ----------
        power : np.ndarray
            A 1D array of power values (e.g., a PSD).
        num_guard_cells : int
            The number of guard cells per side.
        num_average_cells : int
            The number of averaging cells per side.

        Returns
        -------
        np.ndarray
            A 1D array containing the estimated noise floor for each input bin.
        """
        # generate the cfar convolution kernel
        conv = self.gen_cfar_conv(num_guard_cells, num_average_cells)
        # convolve the kernel with IQ data and average by the number of active cells to get noise floor
        return convolve1d(power, conv, mode='wrap') / (2 * num_average_cells)

    def cfar_threshold(self, power, num_guard_cells, num_average_cells):
        """Calculates the adaptive CFAR detection threshold.

        The threshold is computed by scaling the local noise estimate in the
        logarithmic domain.

        Parameters
        ----------
        power : np.ndarray
            A 1D array of power values in decibels (dB).
        num_guard_cells : int
            The number of guard cells per side.
        num_average_cells : int
            The number of averaging cells per side.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - The estimated noise floor in dB.
            - The final adaptive detection threshold in dB.
        """
        # compute the noise
        noise = self.cfar_noise(power, num_guard_cells, num_average_cells)
        # adapt noise in db scale with alpha
        detection_threshold = noise + 10*np.log10(self.cfar_alpha)
        return noise, detection_threshold

    def channelize(self, signal_data: ObsType):
        """Demultiplexes wideband I/Q data into individual channels.

        This method implements a polyphase filterbank or equivalent channelizer
        by frequency-shifting, filtering, and downsampling the input signal.

        Parameters
        ----------
        signal_data : np.ndarray
            Raw I/Q data of shape (2, N).

        Returns
        -------
        np.ndarray
            A 2D complex array of shape (num_channels, samples_per_channel)
            containing the channelized time-domain signals.
        """
        phase_shift_base = -2j * np.pi * self.env.unwrapped.t
        # convert Box signal data into a complex array
        signal_data_step = np.empty(signal_data.shape[1], dtype=np.complex128)
        signal_data_step.real = signal_data[0]
        signal_data_step.imag = signal_data[1]
        sensing_history = []
        channel_data = np.zeros((self.env.unwrapped.num_channels, self.env.unwrapped.samples_per_step //
                                 self.env.unwrapped.num_channels), dtype=complex)
        for k in range(self.env.unwrapped.num_channels):
            # center the freq on the desired bin
            data = signal_data_step * np.exp(phase_shift_base * self.env.unwrapped.fc[k])
            # filter around the center freq
            filtered = signal.lfilter(self.filter, 1.0, data)
            # downsample and store data in channel container
            channel_data[k,:] = filtered[::self.env.unwrapped.num_channels]
        return channel_data

    def process_channel_data(self, channel_data):
        """Applies the CFAR algorithm to the PSD of each channel.

        Parameters
        ----------
        channel_data : np.ndarray
            A 2D complex array of channelized time-domain signals.

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]
            A tuple of lists, where each list contains results for all channels:
            - The Power Spectral Density (PSD) for each channel.
            - The estimated noise floor for each channel's PSD.
            - The adaptive detection threshold for each channel's PSD.
        """
        channel_data_psd = []
        channel_data_noise_floor = []
        channel_data_detection_threshold = []
        for i in range(self.env.unwrapped.num_channels):
            # get the power level for frequencies within each channel
            psd = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(channel_data[i, :])))+1e-6)
            channel_data_psd.append(psd)
            # compute CFAR noise floor and detection threshold
            noise_floor, detection_threshold = self.cfar_threshold(psd, self.num_guard_cells, self.num_average_cells)
            channel_data_noise_floor.append(noise_floor)
            channel_data_detection_threshold.append(detection_threshold)
        return channel_data_psd, channel_data_noise_floor, channel_data_detection_threshold

    def observation(self, signal_data: ObsType) -> WrapperObsType:
        """Executes the full CFAR processing chain to determine channel occupancy.

        Parameters
        ----------
        signal_data : ObsType
            Raw I/Q data from the base environment.

        Returns
        -------
        WrapperObsType
            A binary numpy array indicating the occupancy of each channel.
        """
        # channelize IQ
        channel_data = self.channelize(signal_data)
        # process channel data with CFAR
        cd_psd, cd_noise_floor, cd_detection_threshold = self.process_channel_data(channel_data)
        # need to solidify logic for specifying the number of detections and believe that the bin is occupied
        cd_detections = np.zeros(self.env.unwrapped.num_channels)
        for i in range(len(cd_psd)):
            cd_detections[i]= (cd_psd[i] > cd_detection_threshold).sum()
        # determine a dynamic method to control the threshold value of the number of detections in a time step
        sensing_history = ((cd_detections > self.cfar_detection_threshold)*1).astype(np.int8)
        self.env.unwrapped.info['observation_history'][self.env.unwrapped.info['step_number']] = sensing_history

        return sensing_history


    def observation_testing(self, signal_data: ObsType):
        """A debugging variant of the observation method that returns intermediate results.

        This method exposes the internal states of the CFAR algorithm for analysis
        and visualization.

        Parameters
        ----------
        signal_data : ObsType
            Raw I/Q data from the base environment.

        Returns
        -------
        Tuple[List, List, List, np.ndarray]
            A tuple containing:
            - A list of the PSD arrays for each channel.
            - A list of the noise floor arrays for each channel.
            - A list of the detection threshold arrays for each channel.
            - The final binary sensing history array.
        """
        # channelize IQ
        channel_data = self.channelize(signal_data)
        # process channel data with CFAR
        cd_psd, cd_noise_floor, cd_detection_threshold = self.process_channel_data(channel_data)
        # need to solidify logic for specifying the number of detections and believe that the bin is occupied
        cd_detections = np.zeros(self.env.unwrapped.num_channels)
        for i in range(len(cd_psd)):
            cd_detections[i] = (cd_psd[i] > cd_detection_threshold).sum()
        sensing_history = ((cd_detections > 7) * 1).astype(np.int8)
        self.env.unwrapped.info['observation_history'][self.env.unwrapped.info['step_number']] = sensing_history

        return cd_psd, cd_noise_floor, cd_detection_threshold, sensing_history