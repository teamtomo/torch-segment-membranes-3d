import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

# Import your actual modules
import torch_segment_membranes_3d.utils as utils
from torch_segment_membranes_3d.augment import get_mirrored_img, get_prediction_transforms


class TestUtils:
    """Test utility functions."""
    
    @patch('torch_segment_membranes_3d.utils.gdown.download')
    @patch('torch_segment_membranes_3d.utils.os.makedirs')
    @patch('torch_segment_membranes_3d.utils.os.path.dirname')
    def test_download_model_weights(self, mock_dirname, mock_makedirs, mock_download):
        """Test model weights download."""
        mock_dirname.return_value = "/fake/path"
        
        utils.download_model_weights()
        
        mock_makedirs.assert_called_once()
        mock_download.assert_called_once()
    
    @patch('torch_segment_membranes_3d.utils.os.path.exists')
    @patch('torch_segment_membranes_3d.utils.os.path.dirname')
    def test_get_checkpoint_exists(self, mock_dirname, mock_exists):
        """Test getting checkpoint when it exists."""
        mock_dirname.return_value = "/fake/path"
        mock_exists.return_value = True
        
        result = utils.get_membrain_checkpoint()
        # Use os.path.normpath to handle different path separators across platforms
        assert "membrain_seg_v10.ckpt" in result
        assert "checkpoints" in result
    
    @patch('torch_segment_membranes_3d.utils.download_model_weights')
    @patch('torch_segment_membranes_3d.utils.os.path.exists')
    @patch('torch_segment_membranes_3d.utils.os.path.dirname')
    def test_get_checkpoint_not_exists(self, mock_dirname, mock_exists, mock_download):
        """Test getting checkpoint when it doesn't exist."""
        mock_dirname.return_value = "/fake/path"
        mock_exists.return_value = False
        
        result = utils.get_membrain_checkpoint()
        
        mock_download.assert_called_once()
        # Use os.path.normpath to handle different path separators across platforms
        assert "membrain_seg_v10.ckpt" in result
        assert "checkpoints" in result
    
    def test_fourier_cropping_basic(self):
        """Test basic fourier cropping."""
        data = torch.randn(32, 32, 32)
        new_shape = (16, 16, 16)
        
        result = utils.fourier_cropping_torch(data, new_shape)
        assert result.shape == new_shape
    
    def test_fourier_cropping_with_device(self):
        """Test fourier cropping with specified device."""
        data = torch.randn(16, 16, 16)
        new_shape = (8, 8, 8)
        device = torch.device("cpu")
        
        result = utils.fourier_cropping_torch(data, new_shape, device)
        assert result.shape == new_shape
        assert result.device == device
    
    def test_fourier_extend_basic(self):
        """Test basic fourier extension."""
        data = torch.randn(16, 16, 16)
        new_shape = (32, 32, 32)
        
        result = utils.fourier_extend_torch(data, new_shape)
        assert result.shape == new_shape
    
    def test_fourier_extend_with_device(self):
        """Test fourier extension with specified device."""
        data = torch.randn(8, 8, 8)
        new_shape = (16, 16, 16)
        device = torch.device("cpu")
        
        result = utils.fourier_extend_torch(data, new_shape, device)
        assert result.shape == new_shape
        assert result.device == device

class TestAugmentFunctions:
    """Test augmentation functions."""
    
    def test_get_mirrored_img_no_mirror(self):
        """Test mirrored image with no mirroring (index 0)."""
        img = torch.randn(1, 1, 32, 32, 32)
        
        result = get_mirrored_img(img, 0)
        assert torch.equal(result, img)
    
    def test_get_mirrored_img_various_indices(self):
        """Test different mirroring cases."""
        img = torch.randn(1, 1, 16, 16, 16)
        
        # Test valid mirror indices
        for idx in range(1, 8):
            result = get_mirrored_img(img, idx)
            assert result.shape == img.shape
    
    def test_get_mirrored_img_invalid_index(self):
        """Test invalid mirror index."""
        img = torch.randn(1, 1, 16, 16, 16)
        
        with pytest.raises(AssertionError):
            get_mirrored_img(img, 8)  # Invalid index
        
        with pytest.raises(AssertionError):
            get_mirrored_img(img, -1)  # Invalid index
    
    def test_get_prediction_transforms(self):
        """Test getting prediction transforms."""
        transforms = get_prediction_transforms()
        assert transforms is not None
        # Test that it's a Compose object
        from monai.transforms import Compose
        assert isinstance(transforms, Compose)