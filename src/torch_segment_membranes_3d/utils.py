import torch_segment_membranes_3d as membrane_seg
from typing import Tuple, Union
import gdown, os
    
from scipy.ndimage import distance_transform_edt
from scipy.fft import fftn, ifftn
import torch.fft
import numpy as np
import torch


def download_model_weights():
    """
    Downloads the MemBrain checkpoint either wget or curl.
    """

    download_dir = os.path.join(os.path.dirname(membrane_seg.__file__), "checkpoints")
    os.makedirs(download_dir, exist_ok=True)

    # Correct file ID
    file_id = "1kaN9ihB62OfHLFnyI2_t6Ya3kJm7Wun9"
    output_path = os.path.join(download_dir, "membrain_seg_v10.ckpt")
    url = f"https://drive.google.com/uc?id={file_id}"

    print("Downloading MemBrain weights...")
    gdown.download(url, output_path, quiet=False)
    print("Download complete.")


def get_membrain_checkpoint():
    """
    Get the MemBrain checkpoint.
    """
    checkpoint_path = os.path.join(os.path.dirname(membrane_seg.__file__), "checkpoints", "membrain_seg_v10.ckpt")
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    else:
        download_model_weights()
        return checkpoint_path


def fourier_cropping_torch(
    data: torch.Tensor, new_shape: tuple, device: torch.device = None
) -> torch.Tensor:
    """
    Fourier cropping adapted for PyTorch and GPU, without smoothing functionality.

    Parameters
    ----------
    data : torch.Tensor
        The input data as a 3D torch tensor on GPU.
    new_shape : tuple
        The target shape for the cropped data as a tuple (x, y, z).
    device : torch.device, optional
        The device to use for the computation. If None, the device is set to "cuda" if
        available; otherwise, it is set to "cpu".

    Returns
    -------
    torch.Tensor
        The resized data as a 3D torch tensor.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data = data.to(device)

    # Calculate the FFT of the input data
    data_fft = torch.fft.fftn(data)
    data_fft = torch.fft.fftshift(data_fft)

    # Calculate the cropping indices
    original_shape = torch.tensor(data.shape, device=device)
    new_shape = torch.tensor(new_shape, device=device)
    start_indices = (original_shape - new_shape) // 2
    end_indices = start_indices + new_shape

    # Crop the filtered FFT data
    cropped_fft = data_fft[
        start_indices[0] : end_indices[0],
        start_indices[1] : end_indices[1],
        start_indices[2] : end_indices[2],
    ]

    unshifted_cropped_fft = torch.fft.ifftshift(cropped_fft)

    # Calculate the inverse FFT of the cropped data
    resized_data = torch.real(torch.fft.ifftn(unshifted_cropped_fft))

    return resized_data


def fourier_extend_torch(
    data: torch.Tensor, new_shape: tuple, device: torch.device = None
) -> torch.Tensor:
    """
    Fourier padding adapted for PyTorch and GPU, without smoothing functionality.

    Parameters
    ----------
    data : torch.Tensor
        The input data as a 3D torch tensor on GPU.
    new_shape : tuple
        The target shape for the extended data as a tuple (x, y, z).
    device : torch.device, optional
        The device to use for the computation. If None, the device is set to "cuda" if
        available; otherwise, it is set to "cpu".

    Returns
    -------
    torch.Tensor
        The resized data as a 3D torch tensor.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data = data.to(device)

    data_fft = torch.fft.fftn(data)
    data_fft = torch.fft.fftshift(data_fft)

    padding = [
        (new_dim - old_dim) // 2 for old_dim, new_dim in zip(data.shape, new_shape)
    ]
    padded_fft = torch.nn.functional.pad(
        data_fft,
        pad=[pad for pair in zip(padding, padding) for pad in pair],
        mode="constant",
    )

    unshifted_padded_fft = torch.fft.ifftshift(padded_fft)

    # Calculate the inverse FFT of the cropped data
    resized_data = torch.real(torch.fft.ifftn(unshifted_padded_fft))

    return resized_data