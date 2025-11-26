import numpy as np
import torch

def generate_label_map_from_burst_list(n: int,
                                       time_bins: int,
                                       freq_bins: int,
                                       burst_list: list):
    """
    takes in signal from burst_list and converts to boolean label map
    Generate a binary label map indicating signal presence in time-frequency bins.

    Parameters:
    -----------
    n : int
        Total length of the signal
    time_bins : int
        Number of time bins to divide the signal into
    freq_bins : int
        Number of frequency bins
    burst_list : list of pywaspgen.burst_def.BurstDef object from PyWaspGen
        List of dictionaries containing signal parameters with keys:
        - 'cent_freq': center frequency (normalized)
        - 'bandwidth': signal bandwidth (normalized)
        - 'start': start sample index
        - 'duration': duration in samples
    freq_range : tuple, optional
        Frequency range as (min_freq, max_freq). Default is (-0.5, 0.5)

    Returns:
    --------
    numpy.ndarray
        Binary array of shape (freq_bins, time_bins) where True indicates
        signal presence in that time-frequency bin
    """

    # Initialize the label map
    label_map = np.zeros((freq_bins, time_bins), dtype=bool)

    # Calculate time and frequency bin boundaries
    time_bin_size = n / time_bins
    freq_min, freq_max = -0.5, 0.5  # assumes Normalized frequency about baseband
    freq_bin_size = 1/freq_bins


    # convert objects into list of signals dicts
    signal_dicts = []
    for signal in burst_list[0]:  # assumes it is the default pywasgen wrapper which is a nested list
        signal_dicts.append(vars(signal))
    # Process each signal in the list
    for signal in signal_dicts:
        cent_freq = signal['cent_freq']
        bandwidth = signal['bandwidth']
        start_sample = signal['start']
        duration = signal['duration']

        # Calculate signal frequency boundaries
        sig_freq_min = cent_freq - bandwidth / 2
        sig_freq_max = cent_freq + bandwidth / 2

        # Calculate signal time boundaries
        sig_time_start = start_sample
        sig_time_end = start_sample + duration

        # Find overlapping time bins (use min and max to avoid indexError)
        start_time_bin = max(0, int(sig_time_start // time_bin_size))
        end_time_bin = min(time_bins - 1, int(sig_time_end // time_bin_size))

        # Find overlapping frequency bins
        for freq_idx in range(freq_bins):
            freq_bin_min = freq_min + freq_idx * freq_bin_size
            freq_bin_max = freq_min + (freq_idx + 1) * freq_bin_size

            # Check if signal frequency range overlaps with this frequency bin
            if sig_freq_min < freq_bin_max and sig_freq_max >= freq_bin_min:
                # Mark all overlapping time bins for this frequency bin
                for time_idx in range(start_time_bin, end_time_bin + 1):
                    label_map[freq_idx, time_idx] = True

    return torch.tensor(label_map.T)  # transpose to be time by freq

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self