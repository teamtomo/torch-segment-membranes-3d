from torch_segment_membranes_3d.models.inference_model import PreprocessedSemanticSegmentationUnet
from torch_segment_membranes_3d.augment import get_mirrored_img, get_prediction_transforms
import torch_segment_membranes_3d.utils as utils
from monai.inferers import SlidingWindowInferer
from tqdm import tqdm
import numpy as np
import torch


class MembrainSeg:

    def __init__(self, device=None, sw_batch_size = 4, sw_window_size = 160):

        # Determine Device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set the Sliding Window Parameters
        self.sw_batch_size = sw_batch_size
        self.sw_window_size = sw_window_size

        # Perform sliding window inference on the new data
        roi_size = (sw_window_size, sw_window_size, sw_window_size)
        self.inferer = SlidingWindowInferer(
            roi_size,
            sw_batch_size,
            overlap=0.5,
            progress=False,
            mode="gaussian",
            device=torch.device("cpu"),
        )

        # Get the Model
        checkpoint = utils.get_membrain_checkpoint()

        # Initialize the model and load trained weights from checkpoint
        self.model = PreprocessedSemanticSegmentationUnet.load_from_checkpoint(
            checkpoint,
            map_location=device,
            strict=False,
        )
        self.model.to(device)
        self.model.target_shape = (sw_window_size, sw_window_size, sw_window_size)

        # Put the model into evaluation mode
        self.model.eval()

        # Get Prediction Transforms
        self.transforms = get_prediction_transforms()

    def preprocess(self, data, normalize_data=True):
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
        new_data = self.transforms(new_data)

        # Add batch dimension
        new_data = new_data.unsqueeze(0)

        # Move to device
        new_data = new_data.to('cpu')

        return new_data 

    def run(self, data, threshold=0, test_time_augmentation=True, progress_bar=True):

        # Check input data type for return type matching
        input_is_numpy = isinstance(data, np.ndarray)

        data = self.preprocess(data).to(torch.float32)

        # Perform test time augmentation (8-fold mirroring)
        predictions = torch.zeros_like(data)

        for m in tqdm(range(8 if test_time_augmentation else 1), disable=not progress_bar):
            with torch.no_grad(), torch.cuda.amp.autocast():
                mirrored_input = get_mirrored_img(data.clone(), m).to(self.device)
                mirrored_pred = self.inferer(mirrored_input, self.model)
                if not (isinstance(mirrored_pred, (list, tuple))):
                    mirrored_pred = [mirrored_pred]
                correct_pred = get_mirrored_img(mirrored_pred[0], m)
                predictions += correct_pred.detach().cpu()

        if test_time_augmentation:
            predictions /= 8.0

        # Remove batch and channel dimensions for output
        predictions = predictions.squeeze(0).squeeze(0)

        # Apply segmentation threshold
        predictions[predictions > threshold] = 1
        predictions[predictions <= threshold] = 0

        # Return results
        if input_is_numpy:
            return predictions.numpy()
        else:
            return predictions
