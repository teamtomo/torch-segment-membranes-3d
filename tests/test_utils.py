import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

# Import your actual modules
import torch_segment_membranes_3d.utils as utils
from torch_segment_membranes_3d.augment import get_mirrored_img, get_prediction_transforms


# =============================================================================
# FIXTURES - These cache/reuse expensive objects across tests
# =============================================================================

@pytest.fixture(scope="session")
def sample_torch_tensors():
    """Create sample torch tensors that can be reused."""
    return {
        'small': torch.randn(8, 8, 8),
        'medium': torch.randn(16, 16, 16),
        'large': torch.randn(32, 32, 32),
        'batch': torch.randn(1, 1, 16, 16, 16)
    }

@pytest.fixture
def mock_file_paths():
    """Mock file paths for testing."""
    return {
        'checkpoint': "/fake/path/membrain_seg_v10.ckpt",
        'directory': "/fake/path"
    }


# =============================================================================
# UTILS TESTS
# =============================================================================

class TestUtils:
    """Test utility functions."""
    
    @patch('torch_segment_membranes_3d.utils.gdown.download')
    @patch('torch_segment_membranes_3d.utils.os.makedirs')
    @patch('torch_segment_membranes_3d.utils.os.path.dirname')
    def test_download_model_weights(self, mock_dirname, mock_makedirs, mock_download, mock_file_paths):
        """Test model weights download."""
        mock_dirname.return_value = mock_file_paths['directory']
        
        utils.download_model_weights()
        
        mock_makedirs.assert_called_once()
        mock_download.assert_called_once()
    
    @patch('torch_segment_membranes_3d.utils.os.path.exists')
    @patch('torch_segment_membranes_3d.utils.os.path.dirname')
    def test_get_checkpoint_exists(self, mock_dirname, mock_exists, mock_file_paths):
        """Test getting checkpoint when it exists."""
        mock_dirname.return_value = mock_file_paths['directory']
        mock_exists.return_value = True
        
        result = utils.get_membrain_checkpoint()
        assert "membrain_seg_v10.ckpt" in result
        assert "checkpoints" in result
    
    @patch('torch_segment_membranes_3d.utils.download_model_weights')
    @patch('torch_segment_membranes_3d.utils.os.path.exists')
    @patch('torch_segment_membranes_3d.utils.os.path.dirname')
    def test_get_checkpoint_not_exists(self, mock_dirname, mock_exists, mock_download, mock_file_paths):
        """Test getting checkpoint when it doesn't exist."""
        mock_dirname.return_value = mock_file_paths['directory']
        mock_exists.return_value = False
        
        result = utils.get_membrain_checkpoint()
        
        mock_download.assert_called_once()
        assert "membrain_seg_v10.ckpt" in result
        assert "checkpoints" in result
    
    def test_fourier_cropping_basic(self, sample_torch_tensors):
        """Test basic fourier cropping using fixture data."""
        data = sample_torch_tensors['large']  # 32x32x32
        new_shape = (16, 16, 16)
        
        result = utils.fourier_cropping_torch(data, new_shape)
        assert result.shape == new_shape
    
    def test_fourier_cropping_with_device(self, sample_torch_tensors):
        """Test fourier cropping with specified device."""
        data = sample_torch_tensors['medium']  # 16x16x16
        new_shape = (8, 8, 8)
        device = torch.device("cpu")
        
        result = utils.fourier_cropping_torch(data, new_shape, device)
        assert result.shape == new_shape
        assert result.device == device
    
    def test_fourier_extend_basic(self, sample_torch_tensors):
        """Test basic fourier extension using fixture data."""
        data = sample_torch_tensors['medium']  # 16x16x16
        new_shape = (32, 32, 32)
        
        result = utils.fourier_extend_torch(data, new_shape)
        assert result.shape == new_shape
    
    def test_fourier_extend_with_device(self, sample_torch_tensors):
        """Test fourier extension with specified device."""
        data = sample_torch_tensors['small']  # 8x8x8
        new_shape = (16, 16, 16)
        device = torch.device("cpu")
        
        result = utils.fourier_extend_torch(data, new_shape, device)
        assert result.shape == new_shape
        assert result.device == device


# =============================================================================
# AUGMENTATION TESTS
# =============================================================================

class TestAugmentFunctions:
    """Test augmentation functions."""
    
    def test_get_mirrored_img_no_mirror(self, sample_torch_tensors):
        """Test mirrored image with no mirroring (index 0)."""
        img = sample_torch_tensors['batch']  # 1x1x16x16x16
        
        result = get_mirrored_img(img, 0)
        assert torch.equal(result, img)
    
    def test_get_mirrored_img_various_indices(self, sample_torch_tensors):
        """Test different mirroring cases using fixture data."""
        img = sample_torch_tensors['batch']  # 1x1x16x16x16
        
        # Test valid mirror indices
        for idx in range(1, 8):
            result = get_mirrored_img(img, idx)
            assert result.shape == img.shape
    
    def test_get_mirrored_img_invalid_index(self, sample_torch_tensors):
        """Test invalid mirror index."""
        img = sample_torch_tensors['batch']
        
        with pytest.raises(AssertionError):
            get_mirrored_img(img, 8)  # Invalid index
        
        with pytest.raises(AssertionError):
            get_mirrored_img(img, -1)  # Invalid index
    
    def test_get_prediction_transforms(self):
        """Test getting prediction transforms."""
        transforms = get_prediction_transforms()
        assert transforms is not None
        from monai.transforms import Compose
        assert isinstance(transforms, Compose)