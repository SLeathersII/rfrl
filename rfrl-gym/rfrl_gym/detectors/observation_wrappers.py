from ..envs import RFRLGymIQEnv2
from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar, Union
from gymnasium.spaces import Box, Discrete, Space
from gymnasium import Env, Wrapper
from .detector import Detector
import numpy as np
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
        elif space in ['detect', 'classify']:
            if space == 'detect':
                self.observation_space = Discrete(2**env.unwrapped.num_channels)  # binary for bins
            elif space == 'classify':
                self.observation_space = Discrete((1+env.unwrapped.num_entities)**env.unwrapped.num_channels)

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
        sensing_history = []
        for k in range(self.env.unwrapped.num_channels):
            data = signal_data * np.exp(phase_shift_base*self.env.unwrapped.fc[k])
            filtered = signal.sosfilt(self.sos_filter, data)
            downsampled = filtered[::self.env.unwrapped.num_channels]
            sensed_result = np.sum(np.abs(downsampled)**2.0)\
                                 /(self.env.unwrapped.samples_per_step/self.env.unwrapped.num_channels)
            self.env.unwrapped.info['sensing_energy_history'][self.env.unwrapped.info['step_number']][k] = sensed_result
            # turn the sensed_result into binary based on threshold
            sensing_history.append((sensed_result > self.threshold)*1)
        self.env.unwrapped.info['observation_history'][self.env.unwrapped.info['step_number']] = sensing_history
        return self._observation_space_encoder(sensing_history)

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

    def _observation_space_encoder(self, observation_vect):
        observation_int = 0
        for idx in range(len(observation_vect)):
            observation_int += (self.observation_base ** idx) * observation_vect[idx]
        return observation_int

    def observation(self, signal_data: ObsType) -> WrapperObsType:
        true_history_step = self.env.unwrapped._get_true_step_occupancy()


        self.env.unwrapped.info['observation_history'][self.env.unwrapped.info['step_number']] = true_history_step
        return self._observation_space_encoder(true_history_step)