import numpy as np
import torch

def preprocess(
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