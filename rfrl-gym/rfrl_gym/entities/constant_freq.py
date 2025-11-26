from .entity import Entity


class ConstantFreq(Entity):
    def __init__(self, entity_label, num_channels, channels, onoff=[1,1,0], start=None, stop=None, modem_params=None):
        """
        An entity that always chooses the same action.
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
        """
        super().__init__(entity_label, num_channels, channels, onoff, start, stop, modem_params)

    def _validate_self(self):
        pass

    def _get_action(self):
        if self.on_flag == 0:
            action = -1
        else:
            action = self.channels[0]
        return action

    def _reset(self):
        pass