import numpy as np
from .entity import Entity


class SimpleJammer(Entity):
    def __init__(self, entity_label:str, num_channels: int, channels: list, onoff: list = [1,1,0],
                 start: int = None, stop: int = None, modem_params: dict = None,  avoid_repeats: bool =True):
        """
        A simple jammer which looks at the previous steps occupied frequency bins and random selects one to occupy
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
            avoid_repeats: boolean value determining if jammer can choose the same channel more than once in a row
        """
        self.avoid_repeats = avoid_repeats
        super().__init__(entity_label, num_channels, channels, onoff, start, stop, modem_params)
    
    def _validate_self(self):
        pass       
    
    def _get_action(self):
        if self.on_flag == 0 or self.info['step_number'] == 0:
            return -1
        elif self.on_flag == 1:  
            occupied_channels = np.zeros(self.num_channels)
            if self.info['action_history'][0][self.info['step_number']-1] != -1:
                occupied_channels[self.info['action_history'][0][self.info['step_number']-1]] = 1
            for idx in range(self.num_channels):
                if (self.info['true_history'][self.info['step_number']-1][idx] != self.entity_idx) and (self.info['true_history'][self.info['step_number']-1][idx] != 0):
                    if occupied_channels[idx] == 1 or self.info['true_history'][self.info['step_number']-1][idx] == self.info["num_entities"]+1:
                        if self.current_channel == idx:
                            occupied_channels[idx] = 2
                        else:
                            occupied_channels[idx] = 0
                    else:
                        occupied_channels[idx] = self.info['true_history'][self.info['step_number']-1][idx] > 0
            
            if sum(occupied_channels) == 0:
                return -1
            else:
                if (self.avoid_repeats == True) and (len(np.where(occupied_channels == 1)[0]) > 0):
                    self.current_channel = np.random.choice(np.where(occupied_channels == 1)[0])
                else:
                    self.current_channel = np.random.choice(np.where(occupied_channels > 0)[0])

                return self.current_channel

    def _reset(self):
       self.current_channel = None