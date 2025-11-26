import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import os
from .AbstractDataClass import AbstractDataClass


class Plotter(AbstractDataClass):

    def __init__(self,
                 root_dir: str = None):
        self.root_dir = root_dir
        if isinstance(root_dir, str):
            self.path_container = self.find_h5()
            df = self.data_frame_from_h5() # don't want to locally store all data into object
            self.df = self.summary_df(df)
        else:
            pass  # simply create object with no data wrangling
    def find_h5(self):
        """
        Finds paths to each h5 file from root_dir.
        Returns:
            A list of strings, where each string is the path to h5 file.
        """
        h5_subfolders = []
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.lower().endswith(".h5"):
                    h5_subfolders.append(dirpath)
                    break  # Move to the next directory once a JSON file is found
        # note: if  file structure is certain could use glob.glob('./experiments/*/*/*.h5')
        return [glob.glob(f'{subfolder}/*.h5')[0] for subfolder in h5_subfolders]

    @staticmethod
    def compare_dict(a,b):
        """
        compare input dictionaries and output string with differences for plotting
        """
        track_string = ''
        if a.keys() == b.keys():
            items1 = list(a.values())
            for idx, (key, value) in enumerate(b.items()):
                if value != items1[idx]:
                    #print(idx, key, value, items1[idx])
                    #try:
                    if isinstance(value, (float, np.float64, np.float32)):
                        track_string += f'{key}_{value:.5f}|'
                    else:
                        track_string += f'{key}_{value}|'
                    # except:
                    #     track_string += f'{key}_{value}|'

        if len(track_string) == 0:
            return 'default'
        else:
            return track_string

    def h5_group_to_dict(self,
                         h5_group:h5py.Group):
        """Converts a h5py group to a dictionary specific to Sb3_DataClass.py kwargs."""
        dictionary = {}
        for key, item in h5_group.items():
            if isinstance(item, h5py.Dataset):
                if key == "policy" or key == "device":
                    dictionary[key] = item[()].decode() # undo binary for strings

                else:
                    dictionary[key] = item[()]  # Read dataset into a NumPy array
            elif isinstance(item, h5py.Group):
                dictionary[key] = self.h5_group_to_dict(item)  # Recursive call
        return dictionary

    def data_frame_from_h5(self,
                           path_container: str = None):
        if path_container is None:
            path_container = self.path_container
        hp_short = []
        runs = []
        agents = []
        scenes = []
        total_reward = []
        channel_occupancy = []
        for h5 in path_container:
            file = h5py.File(h5, 'r')
            default_args = self.h5_group_to_dict(file['HyperParameters:0'][
                                                'kwargs'])  # first input should always be default values -- needed to get hp_short strings
            # pull out attribute meta data
            algorithm = file.attrs['Algorithm']
            scene = file.attrs['Scene']
            # go into each run and get hp_short string and data for runs
            for hp in file.keys():
                hp_short_string = self.compare_dict(default_args, self.h5_group_to_dict(file[hp]['kwargs']))
                # go into the individual run data
                for run in file[hp]['run_data']:
                    # pull out each value in run data and pad accumulator lists to match indexes -- need to hard code expected values
                    hp_short.append(
                        hp_short_string)  # consider doing this based on length of accumulators instead outside this loop
                    agents.append(algorithm)
                    scenes.append(scene)
                    total_reward.append(file[hp]['run_data'][run]['cumulative_reward'][()])
                    runs.append(pd.Series(file[hp]['run_data'][run]['reward_history'][()]))
                    #observation_history.append(file[hp]['run_data'][run]['observation_history'][()])
                    #true_history.append(pd.Series(file[hp]['run_data'][run]['true_history'][()]))
                    om = file[hp]['run_data'][run]['true_history'][()]
                    num_channels = om.shape[-1]
                    # get the number of indexes occupied (> 0) and divide by num_channels
                    occupancy = (om>0).sum(axis=1)/num_channels
                    channel_occupancy.append(pd.Series(occupancy))
            file.close()
        df = pd.DataFrame({'reward_history': runs,
                           'total_reward': total_reward,
                           'algorithm': agents,
                           'scene': scenes,
                           'hyper_params_short': hp_short,
                           #'observation_history':observation_history,
                           'channel_occupancy':channel_occupancy}
                           )
        return df

    def _find_group(self,
                    file: h5py.Group,
                    kwargs: dict
                    ):
        """
        needed for abstract class method
        """
        pass
    def _init__setup(self):
        pass
    def save(self):
        pass

    def moving_average_bounds(self,
                              reward_histories: pd.Series = None,
                              window: int = 25):
        """
        Plotting tool to get the upper and lower quantiles for plotting reward moving averages
        :param reward_histories: panda series from reward_history of df object
        :param window: moving window for averaging the reward
        :return: bounds for plotting based on moving averages
        """
        moving_average = np.array([self.reward_moving_average(run, window) for run in reward_histories])
        median = np.quantile(moving_average, .5, axis=0)
        lower_bound = np.quantile(moving_average, .25, axis=0)
        upper_bound = np.quantile(moving_average, .75, axis=0)
        return lower_bound, median, upper_bound

    def plot_all_agents_from_df(self,
                                df: pd.DataFrame = None,
                                reward_moving_averages:list = None,
                                ):
        """
        Function to plot reward moving averages from each agent in df grouped by algorithm, scene, and hyperparameters
        :param df: DataFrame from data_frame_from_h5 method or with same convention
        :param reward_moving_averages: list of moving averages to plot -- Default [25, 50, 100]
        :return: None outputs plots
        """
        if df is None:
            df = self.df

        if reward_moving_averages is None:
            reward_moving_averages = [25, 50, 100]
        sns.set_theme()
        for (algorithm, scene, hp_short), vals in df.groupby(['algorithm', 'scene', 'hyper_parameters']):
            fig, axs = plt.subplots(1, len(reward_moving_averages), figsize=(int(6*len(reward_moving_averages)), 7))
            fig.suptitle(f"Algorithm: {algorithm}\nScene: {scene} ------- Hyper_Params: {hp_short}")
            for i in range(len(reward_moving_averages)):
                axs[i].set_title(f"Reward Moving Average {reward_moving_averages[i]}")
                median = vals.reward_history_median.values[0]
                upper_bound, lower_bound = vals['reward_history_75-25_bounds'].values[0]
                #lower_bound, median, upper_bound = self.moving_average_bounds(vals.reward_history.values, window=reward_moving_averages[i])
                axs[i].plot(median, label='median')
                axs[i].fill_between(np.arange(len(median)), lower_bound, upper_bound, alpha=0.2, label='75-25 quartile')
                axs[i].legend()


    def plot_all_median_rma50(self,
                              df):
        sns.set_theme()
        for (algorithm, scene, hp_short), vals in df.groupby(['algorithm', 'scene', 'hyper_parameters']):
            fig, axs = plt.subplots(1, 2, figsize=(12, 7))
            fig.suptitle(f"Algorithm: {algorithm}\nScene: {scene} ------- Hyper_Params: {hp_short}")
            axs[0].set_title(f"Reward History Median: Moving Average 50")
            median = vals.reward_history_medianRMA50.values[0]
            occupancy_median_run_mean = vals.occupancy_median.values[0].mean()
            # map occupancy to (-1,1) score for randomly selected channels [0 implies random should be 1,
            # 1 implies all channels occupied and the expected score should be -1 if randomly selected
            om = -2*occupancy_median_run_mean.mean()+1  # maps 0 to 1, .5 to 0, and 1 to -1
            upper_bound, lower_bound = vals['reward_history_75-25_boundsRMA50'].values[0]
            # create lines for changes in scene if dynamic
            if "gradual" in scene:
                vertical_line_x_coords = [300, 500, 700, 900, 1100]
            elif 'dynamic' in scene:
                vertical_line_x_coords = [400, 800]
            else:
                vertical_line_x_coords = [None]
            axs[0].plot(median, label='median')
            axs[0].axhline(y=om, xmin=0, xmax=1,
                           label='Random Choice', c='red', ls='--')

            axs[0].vlines(vertical_line_x_coords, ymin=-1.1, ymax=1.1, colors='red',
                          linestyles='dotted', label='scene change')
            axs[0].fill_between(np.arange(len(median)), lower_bound, upper_bound, alpha=0.2, label='75-25 quartile')
            axs[0].legend()
            axs[0].set_ylim(-1.1,1.1)
            axs[1].set_title("Reward Moving Average 50 of Reward History Median Moving Average 50")

            lower_boundRMA = self.reward_moving_average(lower_bound, 50)
            medianRMA = self.reward_moving_average(median, 50)
            upper_boundRMA = self.reward_moving_average(upper_bound, 50)


            axs[1].plot(medianRMA, label='RMA50')
            axs[1].axhline(y=om.mean(), xmin=0, xmax=1,
                           label='Random Choice', c='red', ls='--')
            axs[1].vlines(vertical_line_x_coords, ymin=-1.1, ymax=1.1, colors='red',
                          linestyles='dotted', label='scene change')
            axs[1].fill_between(medianRMA.index,
                                lower_boundRMA, upper_boundRMA, alpha=0.2, label='75-25 quartile RMA')
            axs[1].legend()
            axs[1].set_ylim(-1.1, 1.1)

    def alg_by_scene_df(self,
                        df: pd.DataFrame = None,):
        """
        aggregation tool that takes the summary DataFrame and creates generator object of dataframes
        for each scene paired with and without jammers for each algorithm for its default hyperparameter settings
        and the setting which acheived the largest median reward for the scene
        :param df: DataFrame adhering to rfrl standard
        :return: a list object of aggregated DataFrames to be iterated over for plotting
        """
        if df is None:
            df = self.df
            df = df[~df.scene.str.contains('ij', na=False)]
        df['group_key'] = df['scene'].apply(lambda x: x.split('_ij')[0].split('_sj')[0])
        gen = []
        for group_key, vals in df.groupby(['group_key']):
            cont = []
            for scene, group_vals in vals.groupby('scene'):
                for algorithm, vals2 in group_vals.groupby(['algorithm']):
                    cont.append(vals2[vals2.hyper_parameters == 'default'])
                    # check that default is not the best agent
                    cont.append(vals2[vals2.hyper_parameters != 'default'].nlargest(1, 'median_reward'))
            df_merged = pd.concat(
                cont)  # this gives us the jammer and no jammer scene values for each algorithm with highest reward and default
            gen.append(df_merged)
        return gen

    def plot_alg_by_scene(self,
                          df: pd.DataFrame = None,):
        gen = self.alg_by_scene_df(df)
        for df_merged in gen:
            fig, axs = plt.subplots(1,2,figsize=(13,7))
            df_merged = df_merged.rename(columns={'occupancy_mean': 'occupancy_median'})  # remove later
            for idx, row in df_merged.iterrows():
                idx = 1 if 'sj' in row.scene else 0
                self.plot_df_row(row, axs[idx])

            [a.set_title(f"Reward History Median: Moving Average 50\n{row.scene}") for a in axs]
            [self.plot_scene_context(row, a) for a in axs]
            [a.legend() for a in axs]
            # axs[idx].set_title(f"Reward History Median: Moving Average 50\n{row.scene}")
            # axs[idx].legend()


    @staticmethod
    def plot_df_row(row: pd.Series,
                    axs = None,
                    ):
        """
        Plots rows from an aggregated DataFrame adhering to rfrl data structure
        :param row: Panda series of a row from the DataFrame, (df.iterrows())
        :param axs: matplotlib axis object to plot onto
        :return: the axis which the row has been plotted onto
        """
        if isinstance(axs,type(None)):
            fig, axs = plt.subplots()
        dqn_colors = sns.color_palette('Blues', n_colors=2)
        a2c_colors = sns.color_palette('Oranges', n_colors=2)
        run_median = row.reward_history_medianRMA50
        upper, lower = row['reward_history_75-25_boundsRMA50']
        label = (row.hyper_parameters + ' ' +
                 row.algorithm) if row.hyper_parameters == 'default' else f'Tuned {row.algorithm }'
        if 'dqn' in label.lower():
            c = dqn_colors[0 if row.hyper_parameters == 'default' else 1]
        else:
            c = a2c_colors[0 if row.hyper_parameters == 'default' else 1]
        axs.plot(run_median, label=label, color=c)
        axs.fill_between(np.arange(len(run_median)), lower, upper,
                              alpha=0.2, color=c)
        return axs

    @staticmethod
    def plot_scene_context(row: pd.Series,
                         axs = None):
        """
        Plots scene context from scenarios adhering to rfrl json structures. Context includes location of scene changes
        and the expected reward moving average based on channel occupancy of the scenario
        :param row: Panda series of a row from the DataFrame, (df.iterrows())
        :param axs: matplotlib axis object to plot onto
        :return: the axis which the row has been plotted onto
        """

        if isinstance(axs,type(None)):
            fig, axs = plt.subplots()
        occupancy = row.occupancy_median.mean().mean()
        if "gradual" in row.scene:
            vertical_line_x_coords = [300, 500, 700, 900, 1100]
        elif 'dynamic' in row.scene:
            vertical_line_x_coords = [400, 800]
        else:
            vertical_line_x_coords = [None]
        axs.vlines(vertical_line_x_coords, ymin=-1.1, ymax=1.1, colors='red',
                        linestyles='dotted', label='scene change')
        #axs.set_title(f'Scene: {row.scene}') # set in outter scope
        axs.set_ylim(-1.1, 1.1)
        axs.axhline(y=-2 * occupancy + 1, xmin=0, xmax=1,
                         label='Random Choice', c='red', ls='--')
        return axs

    def plot_hyperparameter(self,
                            hyperparameter: list,
                            df: pd.DataFrame = None,
                            reward_moving_averages: list = None,

                            ):
        """
        Function to plot reward moving averages from each agent in df grouped by input hyperparameter
        :param hyperparameter: list of a chosen hyperparameter strings to plot all runs which look into that hp
        :param df: DataFrame from data_frame_from_h5 method or with same convention
        :param reward_moving_averages: list of moving averages to plot -- Default [25, 50, 100]
        :return: None outputs plots
        """
        if df is None:
            df = self.df
        if reward_moving_averages is None:
            reward_moving_averages = [25, 50, 100]
        sns.set_theme()
        hpset = set(hyperparameter)
        # if 'default' in hpset:
        #     hpset.remove('default')
        for (algorithm, scene), vals in df.groupby(['algorithm', 'scene']):
            # find the mask to input below
            mask = []
            # break up list of hyper_param_short values into list of individual changes
            hyper_param_shorts = []
            for v in vals.hyper_parameters.values:
                hyper_param_shorts.append(list(filter(None, v.split('|'))))
            # isolate the hyper param names and turn to set to get mask
            for i in range(len(hyper_param_shorts)):
                for j in range(len(hyper_param_shorts[i])):
                    hyper_param_shorts[i][j] = hyper_param_shorts[i][j].rsplit('_',1)
                hyper_param_shorts[i] = set([item for sublist in hyper_param_shorts[i] for item in sublist])
            # get mask for grouping by input hyperparameter
            idx = 0
            for param_set in hyper_param_shorts:
                if hpset.issubset(param_set) and (len(hpset) == int(len(param_set)/2)):
                    #print('subset found')
                    mask.append(idx)
                if 'default' in hyperparameter and 'default' in param_set:
                    mask.append(idx)

                idx += 1

            if len(mask) > 0:  # avoid empty plots
                fig, axs = plt.subplots(1, 3, figsize=(17, 7))
                fig.suptitle(f"Algorithm: {algorithm}\nScene: {scene} ----- HP: {hyperparameter}")
                for i in range(len(reward_moving_averages)):
                    axs[i].set_title(f"Reward Moving Average Median {reward_moving_averages[i]}")
                    for hp in np.unique(vals.hyper_parameters.values[mask]):
                        lower_bound, median, upper_bound = self.moving_average_bounds(
                            vals.reward_history.values[vals.hyper_parameters.values == hp], window=reward_moving_averages[i])
                        axs[i].plot(median, label=f'{hp} median')
                        # axs[i].fill_between(np.arange(len(median)), lower_bound, upper_bound, alpha=0.4, label=f'{hp} 75-25 quartile')
                    axs[i].legend()


    def summary_df(self,
                   df: pd.DataFrame = None,
                   threshold:float = 0.80):
        """
        function to create a summary DataFrame which gives summary stats on each run grouped by
        algorithm, scene, and hyper parameter settings
        :param threshold: float value defining threshold for convergence in scene
        :return: DataFrame with summary stats for each run
        """

        algos = []
        scenes = []
        hps = []
        medians = []
        means = []
        rh_medians_bounds = []
        num_runs = []
        fault_rate = []
        reward_history_RMA50 = []
        rh_medians = []
        ttcs = []  # median time until initial convergence
        tottcs = []  # median total time converged
        occupancy_medians = []
        for (alg, sce, hp), vals in df.groupby(['algorithm', 'scene', 'hyper_params_short']):
            agent_reward_history50 = np.array([run.rolling(50).mean().dropna() for run in vals.reward_history.values])
            scene_occupancy_history = np.array([om for om in vals.channel_occupancy])
            num_runs.append(len(agent_reward_history50))
            fault_rate.append(np.mean([(run==0).sum()/len(run) for run in vals.reward_history.values]))
            algos.append(alg)
            scenes.append(sce)
            hps.append(hp)
            medians.append(vals.total_reward.median())
            means.append(vals.total_reward.mean())
            rh_median = np.quantile(agent_reward_history50, .5, axis=0)
            reward_history_RMA50.append(agent_reward_history50)
            #th_medians.append(np.quantile(vals.true_history, .5, axis=0))
            #obv_medians.append(np.quantile(vals.observation_history, .5, axis=0))
            occupancy_medians.append(np.quantile(scene_occupancy_history, .5, axis=0)) # todo: determine if median is worth it (just do a single mean)
            rh_medians.append(rh_median)
            rh_medians_bounds.append((np.quantile(agent_reward_history50, .75, axis=0),
                                      np.quantile(agent_reward_history50, .25, axis=0)))
            filters = pd.Series(rh_median > threshold)
            if (filters.astype(int).groupby((filters != filters.shift()).cumsum()).cumsum() >= 10).any():
                ttcs.append(
                    (filters.astype(int).groupby((filters != filters.shift()).cumsum()).cumsum() >= 10).idxmax() - 9)
            else:
                ttcs.append(None)
            tottcs.append(filters.mean())
        df = pd.DataFrame({'algorithm': algos,
                           'scene': scenes,
                           'hyper_parameters': hps,
                           'num_runs': num_runs,
                           'fault_rate': fault_rate,
                           'time_until_convergence_median': ttcs,
                           'time_converged_median': tottcs,
                           'median_reward': medians,
                           'mean_reward': means,
                           'reward_history_RMA50': reward_history_RMA50,
                           'reward_history_medianRMA50': rh_medians,
                           'reward_history_75-25_boundsRMA50': rh_medians_bounds,
                           "occupancy_median": occupancy_medians})

        return df

