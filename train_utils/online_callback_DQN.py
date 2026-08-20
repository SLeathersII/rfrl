from stable_baselines3.common.callbacks import BaseCallback
import time
import math

class OnlineCallbackDqn(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose: int = 0,
                 render: bool = True,
                 threshold: float = 0.8,
                 tmax: int = 500):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.epsilon_threshold = threshold
        self.tmax = tmax
        self.converged = False  # track when to shift epsilon
        self.TIME = time.time()
        self.render = render


    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        # adjust schedule
        self.model.exploration_schedule = OnlineSchedule(self.model.exploration_initial_eps,
                                                         self.model.exploration_final_eps,
                                                         self.model.exploration_fraction,
                                                         self.epsilon_threshold,
                                                         self.tmax
                                                         )
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        exploration_initial_eps: Initial exploration rate
        exploration_final_eps: Final exploration rate
        exploration_fraction: The fraction of num_timesteps where exploration will reach exploration_final_eps.

        :return: If the callback returns False, training is aborted early.
        """
        # get reward moving average to update OnlineSchedule
        info = self.locals['infos'][0]
        step_number = info['step_number']
        if step_number >= 50:
            rma = info['reward_history'][step_number-50:step_number].mean()
            if rma > self.epsilon_threshold:  # model has met convergence
                if self.model.exploration_schedule.converged: # if already converged
                    pass
                else:
                    self.model.exploration_schedule.converged = True
                    #print(f'Model has converged above threshold: {self.epsilon_threshold} in reward moving average 50')

            else:
                if self.model.exploration_schedule.converged: # if we are in a converged state
                    self.model.exploration_schedule.counter = 0
                    #print(f'Model is no longer converged at threshold value')
                self.model.exploration_schedule.converged = False # kick-start cosine annealing

        if self.render:
            self.training_env.render()
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.

        For off-policy algorithms like SAC, DDPG, TD3 or DQN,
        the notion of rollout corresponds to the steps taken in the environment between two updates.
        """
        # if self.render:
        #     self.training_env.render()

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        #print(self.locals['infos'][0]['reward_history'])
        print(f'Training time: {time.time() - self.TIME}')
        self.reward_history = self.locals['infos'][0]['reward_history']
        pass

class OnlineSchedule:
    """
    OnlineSchedule increases and anneals exploration when reward moving average dips below a prespecified threshold
    This is used in DQN for linearly annealing the exploration fraction
    (epsilon for the epsilon-greedy strategy).

    :param start: value to start with if ``progress_remaining`` = 1
    :param end: value to end with if ``progress_remaining`` = 0
    :param end_fraction: fraction of ``progress_remaining``  where end is reached e.g 0.1
        then end is reached after 10% of the complete training process.
    """

    def __init__(self, start: float,
                 end: float,
                 end_fraction: float,
                 threshold: float = 0.8,
                 tmax: int = 300) -> None:
        self.start = start
        self.end = end
        self.end_fraction = end_fraction
        self.epsilon_threshold = threshold
        self.converged = False # track when to shift epsilon
        self.counter = 0
        self.tmax = tmax


    def __call__(self,
                 progress_remaining: float,
                 #reward_moving_average: float,
                 ) -> float:

        # if (1 - progress_remaining) > self.end_fraction:
        #     self.converged = True
        #     return self.end
        if self.converged:
            return self.end
        else:
            # use cosine annealing schedule to avoid annealing without convergence
            # https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
            exploration_rate = self.end + .5 * (self.start - self.end) * (1 + math.cos((self.counter/self.tmax)*math.pi))
            self.counter += 1
            return exploration_rate

    def __repr__(self) -> str:
        return f"OnlineSchedule(start={self.start}, end={self.end}, end_fraction={self.end_fraction})"

