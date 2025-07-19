from typing import Callable, List, Union

import numpy as np
import torch
from monai.transforms import (
    Compose,
    ToTensor
)

### Hard-coded area

# Hard-coding kernel sizes for pooling operations (work also with smaller network sizes)
pool_op_kernel_sizes = [
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
]  # hard-coded
net_num_pool_op_kernel_sizes = pool_op_kernel_sizes
deep_supervision_scales = [[1, 1, 1]] + [
    list(i) for i in 1 / np.cumprod(np.vstack(net_num_pool_op_kernel_sizes), axis=0)
][:-1]

# Define 3D rotation ranges
data_aug_params = {}
data_aug_params["rotation_x"] = (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi)
data_aug_params["rotation_y"] = (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi)
data_aug_params["rotation_z"] = (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi)

# Which axes should be used for mirroring?
mirror_axes = (0, 1, 2)


def get_mirrored_img(img: torch.Tensor, mirror_idx: int) -> torch.Tensor:
    """
    Get mirrored images for test time augmentation.

    There are 8 possible cases, enumerated from 0 to 7.
    The function supports mirroring across three axes,
    and combinations thereof.

    Parameters
    ----------
    img : torch.Tensor
        Input tensor to be mirrored.
    mirror_idx : int
        Integer index to select mirror case.
        Should be within the range [0, 7] inclusive.

    Returns
    -------
    torch.Tensor
        The mirrored image tensor.

    Raises
    ------
    AssertionError
        If the mirror index is not in the range [0, 7].

    """
    assert mirror_idx < 8 and mirror_idx >= 0
    if mirror_idx == 0:
        return img

    if mirror_idx == 1 and (2 in mirror_axes):
        return torch.flip(img, (4,))

    if mirror_idx == 2 and (1 in mirror_axes):
        return torch.flip(img, (3,))

    if mirror_idx == 3 and (2 in mirror_axes) and (1 in mirror_axes):
        return torch.flip(img, (4, 3))

    if mirror_idx == 4 and (0 in mirror_axes):
        return torch.flip(img, (2,))

    if mirror_idx == 5 and (0 in mirror_axes) and (2 in mirror_axes):
        return torch.flip(img, (4, 2))

    if mirror_idx == 6 and (0 in mirror_axes) and (1 in mirror_axes):
        return torch.flip(img, (3, 2))

    if (
        mirror_idx == 7
        and (0 in mirror_axes)
        and (1 in mirror_axes)
        and (2 in mirror_axes)
    ):
        return torch.flip(img, (4, 3, 2))


def get_prediction_transforms() -> Compose:
    """
    Returns the data augmentation transforms for the prediction phase.

    The function sets up a Compose object containing a transformation for
    converting data to tensors.

    Returns
    -------
    Compose
        A Compose object containing the sequence of transformations for
        the prediction phase.

    """
    transforms = Compose(
        [
            ToTensor(),
        ]
    )
    return transforms