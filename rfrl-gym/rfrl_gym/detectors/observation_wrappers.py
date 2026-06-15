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
    """Modify observations from :meth:`Env.reset` and :meth:`Env.step` using :meth:`observation` function.

    If you would like to apply a function to only the observation before
    passing it to the learning code, you can simply inherit from :class:`ObservationWrapper` and overwrite the method
    :meth:`observation` to implement that transformation. The transformation defined in that method must be
    reflected by the :attr:`env` observation space. Otherwise, you need to specify the new observation space of the
    wrapper by setting :attr:`self.observation_space` in the :meth:`__init__` method of your wrapper.
    """
    # todo specify rfrl_gym to lint the base variables
    def __init__(self, env: Env[ObsType, ActType],
                 space: Union[Space, str],
                 ):
        """Constructor for the observation wrapper.
        Args:
            env: Environment to be wrapped.
            space: gymnasium space for encoding output to policy
        """
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
        """Modifies the :attr:`env` after calling :meth:`reset`, returning a modified observation using :meth:`self.observation`."""
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(
        self, action: ActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`step` using :meth:`self.observation` on the returned observations."""
        signal_data, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(signal_data), reward, terminated, truncated, info

    def observation(self, observation: ObsType) -> WrapperObsType:
        """Returns a modified observation.
        Args:
            observation: The :attr:`env` observation
        Returns:
            The modified observation
        """
        raise NotImplementedError

    def _observation_space_encoder(self, observation_vect):
        """
        Args:
            observation_vect: The modified observation in slow time
        Returns: Encoded observation to be passed to policy
        """
        raise NotImplemented

class EnergyDetector(Sensor):
    """
    rudimentary energy detector class for

    """
    def __init__(self, env: Env[ObsType, ActType],
                 space: Union[Space, str],
                 threshold: float = 0.8,
                 sos_filter: np.ndarray = None
                 ):
        """
        Energy detection based on total energy in channel
        Args:
            env: base environment to wrap
            space: observation space to be encoded to
            threshold: energy threshold for flagging detection
            sos_filter: filter to apply for detection
        """
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
        observation_int = 0
        for idx in range(len(observation_vect)):
            observation_int += (self.observation_base ** idx) * observation_vect[idx]
        return observation_int

    def observation(self, signal_data: ObsType) -> WrapperObsType:
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
        # observation_int = 0
        # for idx in range(len(observation_vect)):
        #     observation_int += (self.observation_base ** idx) * observation_vect[idx]
        # return int(observation_int)
        return self.env.unwrapped._get_true_step_occupancy()

    def observation(self, signal_data: ObsType) -> WrapperObsType:
        true_history_step = self.env.unwrapped._get_true_step_occupancy()
        self.env.unwrapped.info['observation_history'][self.env.unwrapped.info['step_number']] = true_history_step
        return true_history_step # self._observation_space_encoder(true_history_step)


class CA_CFAR(Sensor):

    def __init__(self, env: Env[ObsType, ActType],
                 space: Union[Space, str],
                 p_fa: float,
                 num_guard_cells: int = 3,
                 num_average_cells: int = 12,
                 filter: np.ndarray = None):
        """
        Cell averaging CFAR sensor. Assumes num cells are asymetric and will be doubled around the cell under test (CUT)


        Args:
            env: base environment to be wrapped
            space: space that will be used to define the observation
            p_fa: probability of false alarm (used to derive alpha, alpha = N(p_fa**(-1/N)-1)) for scaling noise floor
            for the detection threshold
            num_guard_cells: number of inner surrounding cells with a value of 0
            num_average_cells: number of outer surrounding cells with a value of 1
            filter: filter to be used for anti-aliasing during channelization
        """
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
        half_conv = np.zeros(num_guard_cells + num_average_cells + 1)
        for avg in range(num_average_cells):
            half_conv[avg] = 1
        return np.concat((half_conv, half_conv[:-1][::-1]))

    def cfar_noise(self, power, num_guard_cells, num_average_cells):
        # generate the cfar convolution kernel
        conv = self.gen_cfar_conv(num_guard_cells, num_average_cells)
        # convolve the kernel with IQ data and average by the number of active cells to get noise floor
        return convolve1d(power, conv, mode='wrap') / (2 * num_average_cells)

    def cfar_threshold(self, power, num_guard_cells, num_average_cells):
        # compute the noise
        noise = self.cfar_noise(power, num_guard_cells, num_average_cells)
        # adapt noise in db scale with alpha
        detection_threshold = noise + 10*np.log10(self.cfar_alpha)
        return noise, detection_threshold

    def channelize(self, signal_data: ObsType):
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