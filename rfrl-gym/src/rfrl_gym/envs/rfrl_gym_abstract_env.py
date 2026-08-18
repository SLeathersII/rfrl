import gymnasium as gym
import time
import json
import numpy as np
from PyQt6.QtWidgets import QApplication
import rfrl_gym.renderers
import rfrl_gym.entities

class RFRLGymAbstractEnv(gym.Env):
    """
Abstract reinforcement learning environment for Radio Frequency (RF) scenarios using Gymnasium.

This class loads scenario configurations from a specified JSON file to initialize 
the environment's channels, entities, rendering, and action/observation spaces.

Parameters
----------
scenario_filename : str
    The filename of the JSON scenario configuration file. This file must be located 
    in the 'scenarios/' directory relative to the execution path. See `Notes` for 
    the detailed structure of this JSON file.
num_episodes : int, optional
    The number of episodes to run in the environment. Default is 1.

Attributes
----------
num_episodes : int
    Number of episodes for the environment run.
scenario_metadata : dict
    The raw parsed JSON dictionary containing all scenario configurations.
num_channels : int
    Number of available RF channels in the environment.
max_steps : int
    The maximum number of steps allowed per episode.
observation_mode : {'detect', 'classify'}
    The type of observations returned to the agent.
reward_mode : {'dsa', 'jam'}
    The reward structure used for the agent.
target_entity : str
    The dictionary key/label of the target entity from the scenario file.
entity_list : list
    A list of initialized entity objects instantiated from `rfrl_gym.entities`.
target_idx : int
    The internal numerical index assigned to the target entity.
num_entities : int
    The total number of entities instantiated in the environment.
render_mode : {'null', 'terminal', 'pyqt'}
    The method used for rendering the environment.
render_fps : int
    Frames per second to use when rendering the environment.
next_frame_time : int
    Internal tracker for rendering timing.
pyqt_app : PyQt5.QtWidgets.QApplication
    The PyQt application instance, initialized only if `render_mode` is 'pyqt'.
action_space : gym.spaces.Discrete
    The discrete action space, sized `1 + num_channels`.
observation_base : int
    The base value used to calculate the observation space size (2 for 'detect', 
    `1 + num_entities` for 'classify').
observation_space : gym.spaces.Discrete
    The discrete observation space, sized `observation_base ** num_channels`.

Notes
-----
The `scenario_filename` parameter points to a JSON file that must contain the following 
hierarchical structure and keys:

* **`environment`** (dict): Contains global environment parameters.
    * `num_channels` (int): Number of RF channels available in the environment.
    * `max_steps` (int): Maximum number of steps allowed per episode.
    * `observation_mode` (str): Must be either 'detect' or 'classify'.
    * `reward_mode` (str): Must be either 'dsa' or 'jam'.
    * `target_entity` (str): The string identifier of the target entity. This must 
      exactly match one of the keys defined in the `entities` block.

* **`entities`** (dict): A dictionary where each key is an entity's label (str), and 
  the value is a nested dictionary of parameters for that specific entity.
    * `type` (str): The exact class name of the entity as it appears in 
      the `rfrl_gym.entities` module (e.g., 'PrimaryUser', 'Jammer').
    * `[additional parameters]`: Any other keys provided in this nested dictionary 
      will be parsed and passed as keyword arguments (`kwargs`) directly to the 
      entity's constructor.

* **`render`** (dict): Contains rendering configurations for the Gym environment.
    * `render_mode` (str): Must match one of the supported modes: 'null', 'terminal', or 'pyqt'.
    * `render_fps` (int): Target frames per second for the renderer.
"""

    metadata = {'render_modes': ['null', 'terminal', 'pyqt'], 'render_fps':4,
                    'reward_modes': ['dsa', 'jam'],
                    'observation_modes': ['detect', 'classify']}

    def __init__(self, scenario_filename, num_episodes=1):   
        self.num_episodes = num_episodes

        # Load in the JSON scenario file and check for valid entries.
        f_idx = open('scenarios/' + scenario_filename)
        self.scenario_metadata = json.load(f_idx)
        self.__validate_scenario_metadata()
        
        # Get the environment parameters from the scenario file.
        self.num_channels = self.scenario_metadata['environment']['num_channels']
        self.max_steps = self.scenario_metadata['environment']['max_steps']
        self.observation_mode = self.scenario_metadata['environment']['observation_mode']
        self.reward_mode = self.scenario_metadata['environment']['reward_mode']
        self.target_entity = self.scenario_metadata['environment']['target_entity']

        # Get the entity parameters from the scenario file and initialize the entities.
        entity_idx = 0
        self.entity_list = []
        for entity in self.scenario_metadata['entities']:  
            entity_idx += 1
            obj_str = 'rfrl_gym.entities.' + self.scenario_metadata['entities'][entity]['type'] + '(entity_label=\'' + str(entity) + '\', num_channels=' + str(self.num_channels) + ', '
            for param in self.scenario_metadata['entities'][entity]:
                if not param == 'type':
                    obj_str += (param + '=' + str(self.scenario_metadata['entities'][entity][param]) + ', ')
            obj_str += ')'
            self.entity_list.append(eval(obj_str))      
            self.entity_list[-1].set_entity_index(entity_idx)
            if entity == self.target_entity:
                self.target_idx = entity_idx
        self.num_entities = len(self.entity_list)

        # Get the render parameters from the scenario file and initialize the render if necessary.
        self.render_mode = self.scenario_metadata['render']['render_mode']
        self.render_fps = self.scenario_metadata['render']['render_fps']
        self.next_frame_time = 0

        if self.render_mode == 'pyqt':
            self.pyqt_app = QApplication([])

        # Set the gym's valid action and observation spaces.
        self.action_space = gym.spaces.Discrete(1+self.num_channels)     
        if self.observation_mode == 'detect':
            self.observation_base = 2
        elif self.observation_mode == 'classify':
            self.observation_base = 1+self.num_entities
        self.observation_space = gym.spaces.Discrete(self.observation_base**self.num_channels)

    def step(self, action):
        action -= 1
        self.info['step_number'] += 1
        self.info['action_history'][0][self.info['step_number']] = action

        # Get entity actions and determine player observation.
        self.info['true_history'][self.info['step_number']], self.info['observation_history'][self.info['step_number']] = self.__get_entity_actions_and_observation()     

        # Calculate the player reward.
        if action == -1:
            self.info['reward_history'][self.info['step_number']] = 0
        else:
            if self.reward_mode == 'dsa':
                self.info['reward_history'][self.info['step_number']] = 2.0*int(self.info['true_history'][self.info['step_number']][action]==0)-1.0
            elif self.reward_mode == 'jam':
                self.info['reward_history'][self.info['step_number']] = 2.0*int(self.info['true_history'][self.info['step_number']][action]==self.target_idx)-1.0
        self.info['cumulative_reward'][self.info['step_number']] = np.sum(self.info['reward_history'])
              
        # Update return variables and run the render.
        observation = self.__observation_space_encoder(self.info['observation_history'][self.info['step_number']])
        reward = self.info['reward_history'][self.info['step_number']]
        done = False
        if self.info['step_number'] == self.max_steps:
            self.info['episode_reward'] = np.append(self.info['episode_reward'], self.info['cumulative_reward'][self.info['step_number']])
            done = True

        return int(observation), reward, done, done, self.info

    def reset(self, options={'reset_type':'soft'}, seed=None):
        # Temporarily store episode specific variables if they exist.

        if options == None:
            episode_number = -1
            episode_reward = np.array([], dtype=float)
            if self.render_mode == 'terminal':
                self.renderer = rfrl_gym.renderers.terminal_renderer.TerminalRenderer(self.num_episodes,
                                                                                      self.scenario_metadata)
            if self.render_mode == 'pyqt':
                self.renderer = rfrl_gym.renderers.pyqt_renderer.PyQtRenderer(self.num_episodes,
                                                                              self.scenario_metadata,
                                                                              mode='abstract',
                                                                              samples_per_step=10000)
            if self.render_mode != 'null':
                self.renderer.reset()
        elif hasattr(self, 'info') and options['reset_type'] == 'soft':
            episode_number = self.info['episode_number']
            episode_reward = self.info['episode_reward']         
        else:
            print('Bad Case')

        
        # Reset the gym info dictionary and if necessary restore episode variables.
        self.info = {}
        self.info['step_number'] = 0
        self.info['num_entities'] = self.num_entities
        self.info['num_episodes'] = self.num_episodes
        self.info['episode_reward'] = episode_reward  
        self.info['episode_number'] = episode_number + 1   
        self.info['action_history'] = -1+np.zeros((self.num_entities+1, self.max_steps+1), dtype=int)
        self.info['true_history'] = np.zeros((self.max_steps+1, self.num_channels), dtype=int)
        self.info['observation_history'] = np.zeros((self.max_steps+1, self.num_channels), dtype=int)
        self.info['reward_history'] = np.zeros(self.max_steps+1, dtype=float)
        self.info['cumulative_reward'] = np.zeros(self.max_steps+1, dtype=float)

        # Reset entities, get initial entity actions, and determine initial player observation.
        for entity in self.entity_list:
            entity.reset(self.info)            
        self.info['true_history'][0], self.info['observation_history'][0] = self.__get_entity_actions_and_observation()

        # Set return variables.  
        observation = self.__observation_space_encoder(self.info['observation_history'][0])
        return int(observation), {}
        
    def render(self):
        if self.render_mode != 'null':
            if self.info['step_number'] == 0:
                self.next_frame_time = time.time()
            
            self.renderer.render(self.info)  
            self.next_frame_time += 1.0/self.render_fps
            time.sleep(1/self.render_fps)

            if time.time() < self.next_frame_time:
                time.sleep(self.next_frame_time - time.time())
        return

    def close(self):        
        input('Press Enter to end the simulation...')
        return

    def __validate_scenario_metadata(self):
        # Validate scenario environment parameters.
        assert self.scenario_metadata['environment']['num_channels'] > 0, "Environment parameter 'num_channels'\
                                                                                                            is invalid."
        assert self.scenario_metadata['environment']['max_steps'] > 0, "Environment parameter 'max_steps' is invalid."
        assert self.scenario_metadata['environment']['observation_mode'] in self.metadata['observation_modes'], \
            f'Invalid observation mode. Must be one of the following options: {self.metadata["observation_modes"]}'
        assert self.scenario_metadata['environment']['reward_mode'] in self.metadata['reward_modes'], \
            f'Invalid reward mode. Must be one of the following options: {self.metadata["reward_modes"]}'
        if self.scenario_metadata['environment']['reward_mode'] == 'jam':
            assert self.scenario_metadata['environment']['target_entity'] in \
                   self.scenario_metadata['entities'].keys() or \
                   self.scenario_metadata['environment']['target_entity'] == None, 'Invalid target entity name.\
                            Must correspond to the name of one of the entity labels in the scenario file.'

        # Validate scenario render parameters.
        assert self.scenario_metadata['render']['render_mode'] is None or \
               self.scenario_metadata['render']['render_mode'] in self.metadata['render_modes'], \
            f'Invalid render mode. Must be one of the following options: {self.metadata["render_modes"]}'
        assert self.scenario_metadata['render']['render_fps'] > 0, "Render parameter 'render_fps' is invalid."
        assert self.scenario_metadata['render']['render_history'] > 0, "Render parameter 'render_history' is invalid."
    
    def __observation_space_encoder(self, observation_vect):
        observation_int = 0
        for idx in range(len(observation_vect)):
            observation_int += (self.observation_base**idx)*observation_vect[idx]

        return observation_int
    
    def __get_entity_actions_and_observation(self):
        # Get each entities actions and determine the observation space.
        entity_idx = 0
        true_observation = np.zeros(self.num_channels, dtype=int)
        for entity in self.entity_list:
            entity_idx += 1
            entity_action = entity.get_action(self.info)
            self.info['action_history'][entity_idx][self.info['step_number']] = entity_action
            # If two or more entities' actions are to choose the same channel, set the observation to the number of entities + 1.
            if entity_action != -1:
                if true_observation[entity_action] == 0:
                    true_observation[entity_action] = entity_idx
                else:
                    true_observation[entity_action] = self.num_entities + 1

        # Set the observation for the player.
        if self.observation_mode == 'detect':
            player_observation = np.array(true_observation > 0, dtype=int)
        elif self.observation_mode == 'classify':
            player_observation = true_observation

        return true_observation, player_observation