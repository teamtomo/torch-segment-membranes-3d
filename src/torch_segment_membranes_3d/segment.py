import torch


class MembrainSeg:

    def __init__(self, device=None):
        self.device = device

        # Get the Model

    def segment(self, data, sw_batch_size = 4, sw_window_size = 160, threshold=0):

        # Segment the Tomogram
        predictions = membrain_seg.membrain_segment(
            data,
            models,
            sw_batch_size=sw_batch_size,
            sw_window_size=sw_window_size,
            test_time_augmentation=True,
            normalize_data=True,
            segmentation_threshold=threshold,
        )

        return predictions
