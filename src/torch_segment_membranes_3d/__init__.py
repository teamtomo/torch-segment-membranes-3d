"""Semantic segmentation of membranes in 3D cryo-ET volumes"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-segment-membranes-3d")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "TeamTomo"
__email__ = "alisterburt@gmail.com"

from torch_segment_membranes_3d.segment import MembrainSeg

def predict_membrane_probabilities(
    data, 
    device=None, 
    sw_batch_size=4, 
    sw_window_size=160,
    test_time_augmentation=True, 
    progress_bar=True
):
    """
    Predict membrane probabilities from 3D cryo-ET data.
    
    Parameters
    ----------
    data : numpy.ndarray or torch.Tensor
        Input 3D volume data
    device : torch.device, optional
        Device to run inference on. Defaults to CUDA if available, else CPU
    sw_batch_size : int, default=4
        Sliding window batch size
    sw_window_size : int, default=160
        Sliding window size
    test_time_augmentation : bool, default=True
        Whether to use 8-fold mirroring augmentation
    progress_bar : bool, default=True
        Whether to show progress bar
        
    Returns
    -------
    numpy.ndarray or torch.Tensor
        Probability map with same type as input
    """
    model = MembrainSeg(
        device=device, 
        sw_batch_size=sw_batch_size, 
        sw_window_size=sw_window_size
    )
    return model.predict_probabilities(
        data, 
        test_time_augmentation=test_time_augmentation, 
        progress_bar=progress_bar
    )


def predict_membrane_mask(
    data, 
    threshold=0, 
    device=None, 
    sw_batch_size=4, 
    sw_window_size=160,
    test_time_augmentation=True, 
    progress_bar=True
):
    """
    Predict binary membrane mask from 3D cryo-ET data.
    
    Parameters
    ----------
    data : numpy.ndarray or torch.Tensor
        Input 3D volume data
    threshold : float, default=0
        Threshold for binarizing probabilities
    device : torch.device, optional
        Device to run inference on. Defaults to CUDA if available, else CPU
    sw_batch_size : int, default=4
        Sliding window batch size
    sw_window_size : int, default=160
        Sliding window size
    test_time_augmentation : bool, default=True
        Whether to use 8-fold mirroring augmentation
    progress_bar : bool, default=True
        Whether to show progress bar
        
    Returns
    -------
    numpy.ndarray or torch.Tensor
        Binary mask with same type as input
    """
    model = MembrainSeg(
        device=device, 
        sw_batch_size=sw_batch_size, 
        sw_window_size=sw_window_size
    )
    return model.predict_mask(
        data, 
        threshold=threshold,
        test_time_augmentation=test_time_augmentation, 
        progress_bar=progress_bar
    )

# Expose the main functions in __all__
__all__ = ['predict_membrane_probabilities', 'predict_membrane_mask',]