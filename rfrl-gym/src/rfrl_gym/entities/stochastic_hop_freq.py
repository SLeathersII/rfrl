import numpy as np
from .entity import Entity


class StochasticHopFreq(Entity):
    def __init__(self, entity_label:str, num_channels: int, channels: list, onoff: list = [1,1,0],
                 start: int = None, stop: int = None, modem_params: dict = None, channel_weights=[]):
        """
        An entity that hops in a random pattern over a set of channels where the probability of chosing each channel is given by a user-defined weighting.
        Args:
            entity_label: Name of entity e.g. constant_freq_1
            num_channels: number of discrete channels for the environment
            channels: List of channels that the entity operates within
            onoff: A list of 3 values [x,y,z]
                   * x is a value of 0 or 1 that determines whether the entity starts in on or off position
                   * y is a positive integer that determines how many steps before turning off (if applicable)
                   * z is a positive integer that determines how many steps before turning on (if applicable)
            start: The time step number when the entity will be enabled and start executing.
            stop: The time step number when the entity will be disabled and stop executing.
            modem_params: Dictionary containing the modem parameters for IQ generation
                          * type: modulation type
                          * order: number of symbols
                          * filter: pulse shaping of data
                          * center frequency: given in list [low, high] in baseband [-.5,.5]
                          * bandwidth: float value where 1 occupies entire channel, 3 occupies surrounding channels...
                          * start: float in (0,1) denoting where generation starts within the step
                          * duration: duty cycle (pulse length) float in (0,1)
            channel_weights: A vector of probabilities corresponding to the channels vector that determines the probability of chosing each channel.
        """
        self.channel_weights = channel_weights
        if len(self.channel_weights) == 0:
            self.channel_weights = None #default value is uniform with None (1.0/len(channels))*np.ones(len(channels))
        super().__init__(entity_label, num_channels, channels, onoff, start, stop, modem_params)

    def _validate_self(self):
        if (len(self.channel_weights) != len(self.channels)) or round(np.sum(self.channel_weights),2) != 1.0:
            raise Exception('Parameter \'channel_weights\' of entity \'{}\' is not valid.'.format(self.entity_label))  

    def _get_action(self):
        if self.on_flag == 0:
            return -1
        elif self.on_flag == 1:
            return np.random.choice(self.channels, 1, p=self.channel_weights)

    def _reset(self):
        pass