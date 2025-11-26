import numpy as np
from .entity import Entity


class FixedHopFreq(Entity):
    def __init__(self, entity_label:str, num_channels: int, channels: list, onoff: list = [1,1,0],
                 start: int = None, stop: int = None, modem_params: dict = None, rand_hop: int = 1):
        """
        An entity that hops in a repeating pattern over a set of given channels.
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
            rand_hop: Toggles whether the channels vector is iterated through sequentially or in a repeating random order
        """
        self.rand_hop = rand_hop
        super().__init__(entity_label, num_channels, channels, onoff, start, stop, modem_params)

        #self.rng = np.random.default_rng()
        if self.rand_hop == 1:
            np.random.shuffle(self.channels)
            #self.rng.shuffle(self.channels)
    
    def _validate_self(self):
        assert self.rand_hop in [0,1], 'Parameter \'rand_hop\' of entity \'{}\' is not valid.'.format(self.entity_label)

    def _get_action(self):
        if self.on_flag == 0:
            return -1
        elif self.on_flag == 1:
            if self.count == 0:
                self.channel_index += 1
            if self.channel_index == len(self.channels):
                self.channel_index = 0
            return self.channels[self.channel_index]

    def _reset(self):
        self.channel_index = -1