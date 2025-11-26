import torch
import torch.fft as fft
from typing import Union


def energy_detect_label_map(
        signal_data: torch.Tensor,
        num_time_bins: int,
        num_freq_bins: int,
        energy_threshold_multiplier: float = 5.0,
        occupation_threshold: float = 0.2
) -> torch.Tensor:
    """
    Generate a 2D label map indicating signal presence across time-frequency bins.

    This function performs energy detection on input signal data by:
    1. Dividing the signal into discrete time bins
    2. Computing FFT and energy for each time bin
    3. Determining frequency bin occupation based on energy thresholds
    4. Returning a binary map of signal presence

    Args:
        signal_data (torch.Tensor): 1D tensor containing the input signal samples
        num_time_bins (int): Number of time bins to divide the signal into
        num_freq_bins (int): Number of frequency bins for analysis
        energy_threshold_multiplier (float, optional): Multiplier for median energy
            to set detection threshold. Defaults to 5.0.
        occupation_threshold (float, optional): Fraction of samples in a frequency
            bin that must exceed energy threshold to mark bin as occupied.
            Defaults to 0.2 (20%).

    Returns:
        torch.Tensor: 2D tensor of shape (num_time_bins, num_freq_bins) with
            binary values indicating signal presence (1.0) or absence (0.0)

    Raises:
        TypeError: If signal_data is not a tensor or bins are not integers
        ValueError: If signal_data is not 1D or bin counts are not positive

    Example:
        >>> burst_gen = pywaspgen.BurstDatagen()
        >>> iq_gen = pywaspgen.IQDatagen("configs/default.json")
        >>> rand_iq_data, rand_burst_list_meta = iq_gen.gen_iqdata(burst_gen.gen_burstlist())
        >>> signal = torch.tensor(rand_iq_data[0])
        >>> label_map = energy_detect_label_map(signal, 16, 32)
    """
    # Input validation
    if not isinstance(signal_data, torch.Tensor):
        raise TypeError("signal_data must be a torch.Tensor")

    if not isinstance(num_time_bins, int) or not isinstance(num_freq_bins, int):
        raise TypeError("num_time_bins and num_freq_bins must be integers")

    if signal_data.dim() != 1:
        raise ValueError("signal_data must be a 1D tensor")

    if num_time_bins <= 0 or num_freq_bins <= 0:
        raise ValueError("num_time_bins and num_freq_bins must be positive integers")

    # Container to store label map rows (one per time bin)
    label_map_rows = []

    # Process each time bin
    for time_bin_data in torch.chunk(signal_data, num_time_bins):
        # Compute FFT and energy spectrum
        fft_result = fft.fft(time_bin_data)
        energy_spectrum = torch.abs(fft_result) ** 2

        # Set energy detection threshold based on median energy
        energy_threshold = energy_spectrum.median() * energy_threshold_multiplier
        # apply FFT shift to energy prior to chunking
        energy_spectrum = fft.fftshift(energy_spectrum)
        # Divide energy spectrum into frequency bins and check occupation
        freq_bin_labels = []
        for freq_bin_energy in torch.chunk(energy_spectrum, num_freq_bins, dim=0):
            # Calculate fraction of samples above threshold
            samples_above_threshold = (freq_bin_energy > energy_threshold).float()
            occupation_ratio = samples_above_threshold.mean()

            # Mark bin as occupied if occupation ratio exceeds threshold
            is_occupied = (occupation_ratio > occupation_threshold).float()
            freq_bin_labels.append(is_occupied)

        # Apply FFT shift to center zero frequency and add to container
        freq_bin_tensor = torch.stack(freq_bin_labels)
       # shifted_labels = fft.fftshift(freq_bin_tensor)
        label_map_rows.append(freq_bin_tensor)

    # Stack all time bin rows to create 2D label map
    return torch.stack(label_map_rows)