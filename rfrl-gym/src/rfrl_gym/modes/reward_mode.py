import numpy as np
from ..envs import RFRLGymIQEnv2
from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar
from gymnasium import Env, Wrapper


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")
WrapperObsType = TypeVar("WrapperObsType")


class RewardMode(Wrapper[ObsType, ActType, ObsType, ActType]):
    """A Gymnasium environment wrapper superclass for radio frequency (RF) 
    reinforcement learning reward shaping and objective formulation.

    This wrapper modifies the scalar reward signal returned by the underlying 
    RF environment's :meth:`step` function. It supports the integration of an 
    oracle mode to evaluate the agent's chosen action against complete system 
    ground truth (upper bound of learning dynamics) rather than partial or noisy observation-space 
    sensor metrics.

    Parameters
    ----------
    env : Env[ObsType, ActType]
        The Gymnasium environment instance to be wrapped.
    oracle : bool, default=False
        If ``True``, enables oracle-assisted reward computation using 
        ground-truth state information rather than estimated or sensed metrics.

    Attributes
    ----------
    oracle : bool
        Flag indicating whether ground-truth oracle feedback is active for 
        objective evaluation.

    Notes
    -----
    Subclasses must override the :meth:`reward` method to implement specific 
    RF optimization objectives (e.g., spectral efficiency maximization, 
    SINR threshold penalty, or packet error rate minimization).
    """

    # todo specify rfrl_gym to lint the base variables
    def __init__(self, env: Env[ObsType, ActType], oracle: bool = False):
        """Initialize the RewardMode wrapper.

        Parameters
        ----------
        env : Env[ObsType, ActType]
            The environment to wrap.
        oracle : bool, default=False
            Whether to base the reward function on ground truth parameters.
        """
        Wrapper.__init__(self, env)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment's dynamics using the agent actions.

        Parameters
        ----------
        action : ActType
            The action-space element representing the RF control decision 
            (e.g., power allocation, frequency selection, or modulation scheme).

        Returns
        -------
        observation : ObsType
            The next observation from the environment space.
        reward : SupportsFloat
            The shaped or oracle-modified reward value based on the optimization objective.
        terminated : bool
            Whether the Markov Decision Process (MDP) terminal state is reached.
        truncated : bool
            Whether the episode truncation condition (e.g., time horizon limit) is met.
        info : dict[str, Any]
            Diagnostic auxiliary information regarding the RF environment step.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, self.reward(action), terminated, truncated, info

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        """Compute the modified scalar reward signal for the given action.

        Parameters
        ----------
        action : ActType
            The action executed in the current environment step, utilized 
            alongside observation or ground truth to compute the reinforcement signal.

        Returns
        -------
        SupportsFloat
            The calculated scalar optimization objective value.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement a concrete reward formulation.
        """
        raise NotImplementedError("Each reward mode must implement its own reward method.")


class DSA(RewardMode):
    """Dynamic Spectrum Access (DSA) reward wrapper for opportunistic spectrum allocation.

    This class implements a reward shaping mechanism for an RL agent operating in a 
    cognitive radio network. It maps the agent's chosen frequency bin allocation 
    action to a discrete reward signal based on the presence of co-channel 
    interference (spectral collision) with primary or other secondary users.

    The reward function models a zero-action backoff strategy alongside a binary 
    collision penalty matrix:
    
    * **Backoff Action (-1)**: The agent chooses not to transmit. No spectral footprint 
      is created, and no co-channel interference occurs. Reward is 0.
    * **Successful Transmission (+1)**: The agent selects an unoccupied frequency bin, 
      achieving successful spectrum access.
    * **Spectral Collision (-1)**: The agent selects an occupied frequency bin, resulting 
      in a collision that disrupts the channel and prevents successful packet delivery.

    Parameters
    ----------
    env : Env[ObsType, ActType]
        The Gymnasium environment instance to be wrapped.

    Notes
    -----
    The reward signal is computed by cross-referencing the agent's action with the 
    environment's ground-truth channel occupancy matrix (``true_history``) for the current 
    temporal snapshot (``step_number``). This implementation relies on accessing 
    the unwrapped environment properties to maintain a precise centralized ledger of 
    telemetry statistics, including running reward logs and cumulative reward trajectories.
    """
    def __init__(self, env: Env[ObsType, ActType]):
        """Initialize the DSA reward mode wrapper."""
        super().__init__(env)
        # this introduces a bug it does not pass a pointer
        #self.info = self.env.unwrapped.info

    def reward(self, action: ActType):
        """Calculate the DSA optimization reward signal based on spectrum occupancy.

        Parameters
        ----------
        action : ActType
            The indexed frequency bin chosen by the agent for transmission, 
            or -1 to signify a sensing-only/backoff state.

        Returns
        -------
        SupportsFloat
            The calculated reinforcement signal: 0 for backoff, +1 for collision-free 
            channel access, and -1 for a spectral collision.
        """
        if action == -1:
            reward = 0
        else:
            # boolean = True if agent action is not in occupied place
            reward = int(2.0 * int(self.env.unwrapped.info['true_history'][self.env.unwrapped.info['step_number']][action] == 0) -1.0)
        self.env.unwrapped.info['reward_history'][self.env.unwrapped.info['step_number']] = reward
        self.env.unwrapped.info['cumulative_reward'][self.env.unwrapped.info['step_number']] = np.sum(self.env.unwrapped.info['reward_history'])
        return reward

class Jam(RewardMode):
    """Radio Frequency (RF) electronic attack and denial jamming reward wrapper.

    This class implements an objective function for a cognitive jamming agent 
    engaged in electronic countermeasure (ECM) operations. The objective is to 
    maximize spectral denial by dynamically aligning the agent's jamming signal 
    with the instantaneous operating frequency of a target emitter entity. 
    Target emitter must be set in run scenario file. 

    The reward formulation defines an electronic warfare target-tracking payoff 
    matrix:

    * **Backoff Action (-1)**: The agent enters a passive sensing or energy-saving 
      state. No active jamming waveform is emitted. Reward is 0.
    * **Successful Jamming (+1)**: The agent's chosen frequency bin matches the 
      instantaneous transmission frequency of the target entity, resulting in localized 
      co-channel noise injection and signal-to-interference-plus-noise ratio (SINR) degradation.
    * **Misaligned Jamming (-1)**: The agent transmits on a non-overlapping frequency bin, 
      wasting emission power and failing to disrupt the target's communications.

    Parameters
    ----------
    env : Env[ObsType, ActType]
        The Gymnasium environment instance to be wrapped.

    Notes
    -----
    
    The wrapper cross-references the agent's action with the target entity's 
    historical trajectory profile (``action_history``) mapping to the target's identifier index 
    (``target_entity``) at the precise temporal slice (``step_number``). Telemetry payload arrays 
    inside the unwrapped environment are mutated directly to maintain global coherence 
    of reward logs.
    """
    def __init__(self, env: Env[ObsType, ActType]):
        """Initialize the Jam reward mode wrapper."""
        super().__init__(env)      

    def reward(self, action: ActType):
        """Compute the electronic attack success payoff based on target frequency alignment.

        Parameters
        ----------
        action : ActType
            The indexed frequency bin where the jamming signal energy is concentrated, 
            or -1 to signify a quiet/sensing state.

        Returns
        -------
        SupportsFloat
            The calculated scalar optimization reinforcement signal: 0 for backoff, 
            +1 for successful spectral alignment/collision, and -1 for ineffective jamming.
        """
        if action == -1:
            reward = 0
        else:
            # Extract the current channel index occupied by the targeted communications entity
            target_idx = self.env.unwrapped.info['action_history'][self.env.unwrapped.target_entity][self.env.unwrapped.info['step_number']]
            # Map structural alignment to binary payoff space: True (1) -> +1.0, False (0) -> -1.0 and 
            reward = int(2.0 * (target_idx == action) -1.0)
        
        self.env.unwrapped.info['reward_history'][self.env.unwrapped.info['step_number']] = reward
        self.env.unwrapped.info['cumulative_reward'][self.env.unwrapped.info['step_number']] = np.sum(
            self.env.unwrapped.info['reward_history'])
        return reward