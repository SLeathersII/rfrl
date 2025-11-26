import torch
import torch.fft as fft
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from .utils import AttrDict

def confusion_metrics(y_true, y_pred):
    """
    Fast computation of metrics only (no plotting) for torch tensors.

    Parameters:
    -----------
    y_true : torch.Tensor
        Ground truth label map (boolean)
    y_pred : torch.Tensor
        Predicted label map (boolean)

    Returns:
    --------
    dict
        Dictionary with computed metrics
    """
    y_true = y_true.bool()
    y_pred = y_pred.bool()

    tp = torch.sum(y_true & y_pred).item()
    tn = torch.sum(~y_true & ~y_pred).item()
    fp = torch.sum(~y_true & y_pred).item()
    fn = torch.sum(y_true & ~y_pred).item()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return AttrDict({
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': accuracy, 'precision': precision,
        'recall': recall, 'f1_score': f1_score
    })


def label_map_confusion_matrix(y_true, y_pred, plot=True, device=None, print_map=False):
    """
    Create a simple confusion matrix from two PyTorch tensor label maps.

    Parameters:
    -----------
    y_true : torch.Tensor
        Ground truth label map of shape (time_bins, freq_bins) - boolean tensor
        Each row represents a time step, each column represents a frequency bin
    y_pred : torch.Tensor
        Predicted label map of shape (time_bins, freq_bins) - boolean tensor
        Each row represents a time step, each column represents a frequency bin
    plot : bool
        Whether to display plots
    figsize : tuple
        Figure size for plots (width, height)
    device : str or torch.device
        Device to perform computations on (if None, uses y_true's device)

    Returns:
    --------
    dict
        Dictionary containing confusion matrix values and metrics (as Python scalars)
    """
    # Validate inputs
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    # Ensure tensors are on the same device
    if device is not None:
        y_true = y_true.to(device)
        y_pred = y_pred.to(device)
    elif y_true.device != y_pred.device:
        y_pred = y_pred.to(y_true.device)

    # Convert to boolean tensors if not already
    y_true = y_true.bool()
    y_pred = y_pred.bool()

    # Get dimensions
    time_bins, freq_bins = y_true.shape

    cm = confusion_metrics(y_true, y_pred)

    confusion_matrix = np.array([[cm.tn, cm.fp],
                                 [cm.fn, cm.tp]])

    if print_map:
        # Print results
        print("=" * 50)
        print("TORCH CONFUSION MATRIX RESULTS")
        print("=" * 50)
        print(f"Device: {y_true.device}")
        print(f"Shape: {tuple(y_true.shape)} (time_bins × freq_bins)")
        print(f"Time bins: {time_bins}, Frequency bins: {freq_bins}")
        print(f"Total bins: {int(time_bins*freq_bins)}")
        print()
        print("Confusion Matrix:")
        print("                Predicted")
        print("              No    Yes")
        print(f"True    No  {cm.tn:6d} {cm.fp:6d}")
        print(f"        Yes {cm.fn:6d} {cm.tp:6d}")
        print()
        print("Components:")
        print(f"  True Positives:  {cm.tp:6d}")
        print(f"  True Negatives:  {cm.tn:6d}")
        print(f"  False Positives: {cm.fp:6d}")
        print(f"  False Negatives: {cm.fn:6d}")
        print()
        print("Metrics:")
        print(f"  Accuracy:  {cm.accuracy:.4f}")
        print(f"  Precision: {cm.precision:.4f}")
        print(f"  Recall:    {cm.recall:.4f}")
        print(f"  F1-Score:  {cm.f1_score:.4f}")

    if plot:
        # Convert tensors to numpy for plotting
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()

        # Blue for diagonal (correct predictions), Red for off-diagonal (errors)
        def create_confusion_colormap():
            # Create a matrix to weight the colormap based on position
            weight_matrix = np.array([[1, -1],  # TN (good), FP (bad)
                                      [-1, 1]])  # FN (bad), TP (good)

            # Normalize the confusion matrix values to create the color mapping
            max_val = np.max(confusion_matrix)
            if max_val > 0:
                # Create color values: positive for diagonal, negative for off-diagonal
                color_values = confusion_matrix * weight_matrix / max_val
                return color_values
            else:
                return weight_matrix

        # Plot 1: Confusion Matrix (separate figure)
        plt.figure(figsize=(4, 3))
        # Create the color mapping
        color_values = create_confusion_colormap()

        # Create custom colormap: blue for positive (diagonal), red for negative (off-diagonal)
        colors = ['red', 'white', 'blue']
        n_bins = 256
        custom_cmap = LinearSegmentedColormap.from_list('confusion', colors, N=n_bins)

        im1 = plt.imshow(color_values, cmap=custom_cmap, aspect='equal', vmin=-1, vmax=1)

        # Add text annotations
        negative = cm.tn+cm.fn
        positive = cm.fp+cm.tp
        label_matrix = np.array([[cm.tn/negative, cm.fp/positive],
                                 [cm.fn/negative, cm.tp/positive]])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, f'{label_matrix[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=16, fontweight='bold')

        plt.xticks([0, 1], ['No Signal', 'Signal'])
        plt.yticks([0, 1], ['No Signal', 'Signal'])
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title(f'Confusion Matrix\nF1: {cm.f1_score:.4f}', fontsize=14)

        # Add colorbar
        cbar1 = plt.colorbar(im1, shrink=0.8)
        cbar1.set_label('Count', fontsize=12)
        plt.tight_layout()
        plt.show()

        plt.tight_layout()
        # Plot 2: Ground Truth
        plt.figure(figsize=(10, 6))
        # Transpose to show time on x-axis, frequency on y-axis
        im2 = plt.imshow(y_true_np.T.astype(int), aspect='auto', origin='lower',
                         cmap='binary', extent=[0, time_bins, -0.5, 0.5])
        plt.ylim(-.5,.5)
        plt.xlim(0,time_bins)
        plt.title('Ground Truth')
        plt.xlabel('Time Bins')
        plt.ylabel('Normalized frequency bins [f/F_s]')
        # create grid
        plt.hlines(y=fft.fftshift(fft.fftfreq(freq_bins)),xmin=0,xmax=time_bins)
        plt.vlines(x=range(time_bins), ymin=-.5, ymax=0.5)
        # Set tick locations to match actual bin indices
        plt.xticks(range(time_bins))
        plt.yticks(fft.fftshift(fft.fftfreq(20)))

        plt.colorbar(im2, shrink=0.8, label='Signal Present')
        plt.tight_layout()
        plt.show()
        # Plot 3: Prediction Errors
        plt.figure(figsize=(10, 6))
        # Create error map: 0=correct, 1=false positive, -1=false negative
        error_map = np.zeros_like(y_true_np, dtype=int)
        error_map[~y_true_np & y_pred_np] = 1  # False Positives (red)
        error_map[y_true_np & ~y_pred_np] = -1  # False Negatives (blue)

        # Transpose to show time on x-axis, frequency on y-axis
        im3 = plt.imshow(error_map.T, aspect='auto', origin='lower',
                         cmap='RdBu', vmin=-1, vmax=1, extent=[0, time_bins, -0.5, 0.5])
        plt.ylim(-.5,.5)
        plt.xlim(0,time_bins)
        plt.title('Prediction Errors')
        plt.xlabel('Time Bins')
        plt.ylabel('Frequency Bins')

        # Set tick locations to match actual bin indices
        plt.xticks(range(time_bins))
        plt.yticks(fft.fftshift(fft.fftfreq(20)))
        # create grid
        plt.hlines(y=fft.fftshift(fft.fftfreq(freq_bins)), xmin=0, xmax=time_bins)
        plt.vlines(x=range(time_bins), ymin=-.5, ymax=0.5)
        # Custom colorbar for errors
        cbar3 = plt.colorbar(im3, shrink=0.8, ticks=[-1, 0, 1])
        cbar3.ax.set_yticklabels(['False Neg', 'Correct', 'False Pos'])

        plt.tight_layout()
        plt.show()

    # Return results dictionary (all as Python scalars for easy access)
    cm.confusion_matrix = confusion_matrix
    cm.time_bins = time_bins
    cm.freq_bins = freq_bins

    return cm