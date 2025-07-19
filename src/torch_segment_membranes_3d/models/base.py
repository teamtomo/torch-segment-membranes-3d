from monai.transforms import AsDiscrete, Compose, EnsureType
from monai.networks.nets import DynUNet
import pytorch_lightning as pl
from typing import Tuple
import torch

class SemanticSegmentationUnet(pl.LightningModule):
    """Implementation of a Unet for semantic segmentation.

    This is uses the monai Unet. See the monai docs for more details
    on the parameters.
    https://docs.monai.io/en/stable/networks.html#unet

    Parameters
    ----------
    spatial_dims : int
        The number of spatial dimensions is in the data.
    in_channels : int
        The number of channels in the input tensor.
        The default value is 1.
    out_channels : int
        The number of channels in the output tensor.
        The default value is 1.
    channels : Tuple[int, ...]
        The number of channels in each layer of the encoder/decoder.
        default=[32, 64, 128, 256, 512, 1024]
    strides : Tuple[int, ...], default=(1, 2, 2, 2, 2, 2)
        The strides for the convolutions. Must have len(channels - 1) elements.
    batch_size : int, default=32
        The batch size for training.
    image_key : str
        The value in the batch data dictionary containing the input image.
        Default value is "image".
    label_key : str
        The value in the batch data dictionary containing the labels.
        Default value is "label"
    roi_size : Tuple[int, ...]
        The size of the sliding window for the validation inference.
        Default value is (160, 160, 160).
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: Tuple[int, ...] = [32, 64, 128, 256, 512, 1024],
        strides: Tuple[int, ...] = (1, 2, 2, 2, 2, 2),
        batch_size: int = 32,
        image_key: str = "image",
        label_key: str = "label",
        roi_size: Tuple[int, ...] = (160, 160, 160),
    ):
        super().__init__()

        # store parameters
        self.batch_size = batch_size
        self.image_key = image_key
        self.label_key = label_key
        self.roi_size = roi_size

        # make the network
        self._model = DynUNetDirectDeepSupervision(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3, 3, 3, 3, 3),
            strides=strides,
            upsample_kernel_size=(1, 2, 2, 2, 2, 2),
            filters=channels,
            res_block=True,
            deep_supervision=True,
            deep_supr_num=2,
        )

        self.post_label = Compose(
            [
                EnsureType("tensor", device="cpu"),
                AsDiscrete(to_onehot=2),
            ]
        )

    def forward(self, x) -> torch.Tensor:
        """Implementation of the forward pass.

        See the pytorch-lightning module documentation for details.
        """
        return self._model(x)

    
class DynUNetDirectDeepSupervision(DynUNet):
    """Adjusted DynUNet outputting low-resolution deep supervision images.

    This is in contrast to the original DynUNet implementation: Here, images
    from lower stages are first upsampled, and then compared to the original
    resolution GT image.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """Forward pass."""
        out = self.skip_layers(x)
        out = self.output_block(out)
        if self.deep_supervision:
            out_all = [out]
            for feature_map in self.heads:
                out_all.append(feature_map)
            return out_all
        return out
