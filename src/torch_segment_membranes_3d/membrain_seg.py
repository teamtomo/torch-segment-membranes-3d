import os

import gdown
import numpy as np
import torch
from membrain_seg.segmentation.dataloading.memseg_augmentation import get_mirrored_img, get_prediction_transforms
from membrain_seg.segmentation.networks.inference_unet import (
    PreprocessedSemanticSegmentationUnet,
)
from monai.inferers import SlidingWindowInferer

import copick_torch


def membrain_preprocess(
    data,
    transforms,
    device,
    normalize_data=True,
):
    """
    Preprocess tomogram data from numpy array or PyTorch tensor for inference.

    Adapted from load_data_for_inference in membrain-seg repository.
    """
    # Convert torch tensor to numpy if needed
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    # Normalize data if requested
    if normalize_data:
        mean_val = np.mean(data)
        std_val = np.std(data)
        data = (data - mean_val) / std_val

    # Add channel dimension (C, H, W, D)
    new_data = np.expand_dims(data, 0)

    # Apply transforms
    new_data = transforms(new_data)

    # Add batch dimension
    new_data = new_data.unsqueeze(0)

    # Move to device
    new_data = new_data.to(device)

    return new_data


def membrain_segment(
    data,
    pl_model,
    sw_batch_size=4,
    sw_window_size=160,
    test_time_augmentation=True,
    normalize_data=True,
    segmentation_threshold=0.0,
):
    """
    Segment tomograms using the membrain-seg trained model from in-memory data.

    This function is heavily adapted from the segment() function in the membrain-seg
    repository, modified to work with in-memory numpy arrays or PyTorch tensors
    instead of file paths.

    Parameters
    ----------
    data : np.ndarray or torch.Tensor
        The 3D tomogram data to be segmented.
    sw_window_size: int, optional
        Sliding window size used for inference. Must be a multiple of 32.
    test_time_augmentation: bool, optional
        If True, test-time augmentation is performed.
    normalize_data : bool, optional
        Whether to normalize the input data (default is True).
    segmentation_threshold: float, optional
        Threshold for the membrane segmentation (default: 0.0).

    Returns
    -------
    predictions : torch.Tensor or np.ndarray
        The segmentation predictions (same type as input data).
    """

    # Check input data type for return type matching
    input_is_numpy = isinstance(data, np.ndarray)

    # Get Model Device
    device = pl_model.device

    if sw_window_size % 32 != 0:
        raise OSError("Sliding window size must be multiple of 32!")
    pl_model.target_shape = (sw_window_size, sw_window_size, sw_window_size)

    # Preprocess the data
    transforms = get_prediction_transforms()
    new_data = membrain_preprocess(
        data,
        transforms,
        device=torch.device("cpu"),
        normalize_data=normalize_data,
    )
    new_data = new_data.to(torch.float32)

    # Put the model into evaluation mode
    pl_model.eval()

    # Perform sliding window inference on the new data
    roi_size = (sw_window_size, sw_window_size, sw_window_size)
    inferer = SlidingWindowInferer(
        roi_size,
        sw_batch_size,
        overlap=0.5,
        progress=False,
        mode="gaussian",
        device=torch.device("cpu"),
    )

    # Perform test time augmentation (8-fold mirroring)
    predictions = torch.zeros_like(new_data)

    for m in range(8 if test_time_augmentation else 1):
        with torch.no_grad(), torch.cuda.amp.autocast():
            mirrored_input = get_mirrored_img(new_data.clone(), m).to(device)
            mirrored_pred = inferer(mirrored_input, pl_model)
            if not (isinstance(mirrored_pred, (list, tuple))):
                mirrored_pred = [mirrored_pred]
            correct_pred = get_mirrored_img(mirrored_pred[0], m)
            predictions += correct_pred.detach().cpu()

    if test_time_augmentation:
        predictions /= 8.0

    # Remove batch and channel dimensions for output
    predictions = predictions.squeeze(0).squeeze(0)

    # Apply segmentation threshold
    predictions[predictions > segmentation_threshold] = 1
    predictions[predictions <= segmentation_threshold] = 0

    # Return results
    if input_is_numpy:
        return predictions.numpy()
    else:
        return predictions


def membrane_seg_init(gpu_id: int):
    """
    Initialize the MemBrain segmentation model.

    """

    # Load the trained PyTorch Lightning model
    model_checkpoint = get_membrain_checkpoint()
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # Initialize the model and load trained weights from checkpoint
    pl_model = PreprocessedSemanticSegmentationUnet.load_from_checkpoint(
        model_checkpoint,
        map_location=device,
        strict=False,
    )
    pl_model.to(device)

    # Put the model into evaluation mode
    pl_model.eval()

    return pl_model


def download_model_weights():
    """
    Downloads the MemBrain checkpoint either wget or curl.
    """

    download_dir = os.path.join(os.path.dirname(copick_torch.__file__), "checkpoints")
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
    checkpoint_path = os.path.join(os.path.dirname(copick_torch.__file__), "checkpoints", "membrain_seg_v10.ckpt")
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    else:
        download_model_weights()
        return checkpoint_path
