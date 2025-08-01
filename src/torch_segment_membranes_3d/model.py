from monai.networks.nets import DynUNet
import torch.nn.functional as F
import torch


class DynUNetDirectDeepSupervision(DynUNet):
    """Simplified DynUNet with direct deep supervision."""

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


def create_membrain_model():
    """Factory function to create the MembrainSeg model."""
    return DynUNetDirectDeepSupervision(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        kernel_size=(3, 3, 3, 3, 3, 3),
        strides=(1, 2, 2, 2, 2, 2),
        upsample_kernel_size=(1, 2, 2, 2, 2, 2),
        filters=[32, 64, 128, 256, 512, 1024],
        res_block=True,
        deep_supervision=True,
        deep_supr_num=2,
    )


def load_model_from_checkpoint(checkpoint_path, device=None):
    """Load model with weights from checkpoint."""
    model = create_membrain_model()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove prefixes if needed
        model_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_model.'):
                model_state_dict[key[7:]] = value
            else:
                model_state_dict[key] = value
    else:
        model_state_dict = checkpoint
        
    model.load_state_dict(model_state_dict, strict=False)
    return model