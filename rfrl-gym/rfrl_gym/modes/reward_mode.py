import numpy as np
from ..envs import RFRLGymIQEnv2
from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar
from gymnasium import Env, Wrapper


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")
WrapperObsType = TypeVar("WrapperObsType")


class RewardMode(Wrapper[ObsType, ActType, ObsType, ActType]):
    """Superclass of wrappers that can modify the returning reward from a step.
    Passes in the action from the current step and checks against the ground truth to determine reward signal
    """

    # todo specify rfrl_gym to lint the base variables
    def __init__(self, env: Env[ObsType, ActType]):
        """Constructor for the Reward wrapper.
        Args:
            env: Environment to be wrapped.
        """
        Wrapper.__init__(self, env)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, self.reward(action), terminated, truncated, info

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        """Returns a modified environment ``reward``.
        Args:
            reward: The :attr:`env` :meth:`step` reward
        Returns:
            The modified `reward`
        """
        raise NotImplementedError


class DSA(RewardMode):
    """
    dynamic spectrum access reward mode. Gives a reward of +1 if no spectral collision and reward of -1
    if the RL agent overlaps in frequency bin
    """
    def __init__(self, env: Env[ObsType, ActType]):
        super().__init__(env)
        # this introduces a bug it does not pass a pointer
        #self.info = self.env.unwrapped.info

    def reward(self, action: ActType):
        if action == -1:
            reward = 0
        else:
            # boolean = True if agent action is not in occupied place
            reward = int(2.0 * int(self.env.unwrapped.info['true_history'][self.env.unwrapped.info['step_number']][action] == 0) -1.0)
        self.env.unwrapped.info['reward_history'][self.env.unwrapped.info['step_number']] = reward
        self.env.unwrapped.info['cumulative_reward'][self.env.unwrapped.info['step_number']] = np.sum(self.env.unwrapped.info['reward_history'])
        return reward

class Jam(RewardMode):
    """
    Jam mode gives a reward if the agent transmits in the same frequency bin as the target entity
    """
    def __init__(self, env: Env[ObsType, ActType]):
        super().__init__(env)
        # create info pointer locally for brevity

    def reward(self, action: ActType):
        if action == -1:
            reward = 0
        else:
            # get target idx from action_history
            target_idx = self.env.unwrapped.info['action_history'][self.env.unwrapped.target_entity][self.env.unwrapped.info['step_number']]
            reward = int(2.0 * (target_idx == action) -1.0)
        self.env.unwrapped.info['reward_history'][self.env.unwrapped.info['step_number']] = reward
        self.env.unwrapped.info['cumulative_reward'][self.env.unwrapped.info['step_number']] = np.sum(
            self.env.unwrapped.info['reward_history'])
        return reward