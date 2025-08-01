import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from torch_segment_membranes_3d.segment import MembrainSeg


# =============================================================================
# FIXTURES - These cache/reuse expensive objects across tests
# =============================================================================

@pytest.fixture(scope="session")
def sample_3d_data():
    """Create sample 3D data that can be reused across tests."""
    return {
        'small': np.random.randn(8, 8, 8),
        'medium': np.random.randn(16, 16, 16),
        'large': np.random.randn(32, 32, 32)
    }

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
def mock_checkpoint_path():
    """Mock checkpoint path."""
    return "/fake/checkpoint.ckpt"

@pytest.fixture
def mocked_membrain_seg():
    """Create a mocked MembrainSeg instance for testing."""
    with patch('torch_segment_membranes_3d.segment.utils.get_membrain_checkpoint'), \
         patch('torch_segment_membranes_3d.segment.load_model_from_checkpoint') as mock_load, \
         patch('torch_segment_membranes_3d.segment.get_prediction_transforms'), \
         patch('torch_segment_membranes_3d.segment.SlidingWindowInferer'):
        
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        seg = MembrainSeg()
        seg.transforms = MagicMock()
        return seg

@pytest.fixture
def mock_transforms():
    """Mock transforms that return predictable shapes."""
    def _mock_transform(data):
        # Return tensor with same spatial dims but add channel
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float().unsqueeze(0)
        return data.unsqueeze(0) if len(data.shape) == 3 else data
    
    return MagicMock(side_effect=_mock_transform)


# =============================================================================
# MEMBRAIN SEG TESTS
# =============================================================================

class TestMembrainSeg:
    """Tests for MembrainSeg class using fixtures."""
    
    @patch('torch_segment_membranes_3d.segment.utils.get_membrain_checkpoint')
    @patch('torch_segment_membranes_3d.segment.load_model_from_checkpoint')
    @patch('torch_segment_membranes_3d.segment.get_prediction_transforms')
    @patch('torch_segment_membranes_3d.segment.SlidingWindowInferer')
    def test_init(self, mock_inferer, mock_transforms, mock_load_checkpoint, mock_get_checkpoint, mock_checkpoint_path):
        """Test MembrainSeg initialization."""
        mock_get_checkpoint.return_value = mock_checkpoint_path
        mock_model = MagicMock()
        mock_load_checkpoint.return_value = mock_model
        mock_transforms.return_value = MagicMock()
        
        seg = MembrainSeg()
        
        assert seg.sw_batch_size == 4
        assert seg.sw_window_size == 160
        mock_model.eval.assert_called_once()
        mock_load_checkpoint.assert_called_once_with(mock_checkpoint_path, None)
    
    @patch('torch_segment_membranes_3d.segment.utils.get_membrain_checkpoint')
    @patch('torch_segment_membranes_3d.segment.load_model_from_checkpoint')
    @patch('torch_segment_membranes_3d.segment.get_prediction_transforms')
    @patch('torch_segment_membranes_3d.segment.SlidingWindowInferer')
    def test_init_custom_params(self, mock_inferer, mock_transforms, mock_load_checkpoint, mock_get_checkpoint):
        """Test MembrainSeg initialization with custom parameters."""
        mock_get_checkpoint.return_value = "/fake/checkpoint.ckpt"
        mock_model = MagicMock()
        mock_load_checkpoint.return_value = mock_model
        mock_transforms.return_value = MagicMock()
        
        seg = MembrainSeg(sw_batch_size=8, sw_window_size=128)
        
        assert seg.sw_batch_size == 8
        assert seg.sw_window_size == 128
    
    @patch('torch_segment_membranes_3d.segment.utils.get_membrain_checkpoint')
    @patch('torch_segment_membranes_3d.segment.load_model_from_checkpoint')
    @patch('torch_segment_membranes_3d.segment.get_prediction_transforms')
    @patch('torch_segment_membranes_3d.segment.SlidingWindowInferer')
    def test_init_custom_checkpoint(self, mock_inferer, mock_transforms, mock_load_checkpoint, mock_get_checkpoint):
        """Test MembrainSeg initialization with custom checkpoint."""
        custom_checkpoint = "/custom/checkpoint.ckpt"
        mock_model = MagicMock()
        mock_load_checkpoint.return_value = mock_model
        mock_transforms.return_value = MagicMock()
        
        seg = MembrainSeg(checkpoint=custom_checkpoint)
        
        # Should not call get_membrain_checkpoint when checkpoint is provided
        mock_get_checkpoint.assert_not_called()
        mock_load_checkpoint.assert_called_once_with(custom_checkpoint, None)
    
    def test_preprocess_numpy(self, mocked_membrain_seg, mock_transforms, sample_3d_data):
        """Test preprocessing numpy array using fixtures."""
        mocked_membrain_seg.transforms = mock_transforms
        
        data = sample_3d_data['large']  # 32x32x32
        result = mocked_membrain_seg.preprocess(data)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1  # batch dimension
    
    def test_preprocess_torch(self, mocked_membrain_seg, mock_transforms, sample_torch_tensors):
        """Test preprocessing torch tensor using fixtures."""
        mocked_membrain_seg.transforms = mock_transforms
        
        data = sample_torch_tensors['medium']  # 16x16x16
        result = mocked_membrain_seg.preprocess(data)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1
    
    def test_preprocess_no_normalization(self, mocked_membrain_seg, mock_transforms, sample_3d_data):
        """Test preprocessing without normalization."""
        mocked_membrain_seg.transforms = mock_transforms
        
        data = sample_3d_data['small']  # 8x8x8
        result = mocked_membrain_seg.preprocess(data, normalize_data=False)
        
        assert isinstance(result, torch.Tensor)
    
    @patch('torch_segment_membranes_3d.segment.get_mirrored_img')
    @patch('torch_segment_membranes_3d.segment.tqdm')
    def test_predict_probabilities_numpy(self, mock_tqdm, mock_mirrored, mocked_membrain_seg, sample_3d_data):
        """Test predict_probabilities method with numpy input using fixtures."""
        data = sample_3d_data['medium']  # 16x16x16
        
        mocked_membrain_seg.preprocess = MagicMock(return_value=torch.randn(1, 1, 16, 16, 16))
        mocked_membrain_seg.inferer = MagicMock(return_value=torch.randn(1, 1, 16, 16, 16))
        mock_mirrored.side_effect = lambda x, m: x
        mock_tqdm.side_effect = lambda x, disable=False: x
        
        result = mocked_membrain_seg.predict_probabilities(data, test_time_augmentation=False)
        
        assert isinstance(result, np.ndarray)
    
    @patch('torch_segment_membranes_3d.segment.get_mirrored_img')
    @patch('torch_segment_membranes_3d.segment.tqdm')
    def test_predict_probabilities_torch(self, mock_tqdm, mock_mirrored, mocked_membrain_seg, sample_torch_tensors):
        """Test predict_probabilities method with torch input using fixtures."""
        data = sample_torch_tensors['small']  # 8x8x8
        
        mocked_membrain_seg.preprocess = MagicMock(return_value=torch.randn(1, 1, 8, 8, 8))
        mocked_membrain_seg.inferer = MagicMock(return_value=torch.randn(1, 1, 8, 8, 8))
        mock_mirrored.side_effect = lambda x, m: x
        mock_tqdm.side_effect = lambda x, disable=False: x
        
        result = mocked_membrain_seg.predict_probabilities(data, test_time_augmentation=False)
        
        assert isinstance(result, torch.Tensor)
    
    @patch('torch_segment_membranes_3d.segment.get_mirrored_img')
    @patch('torch_segment_membranes_3d.segment.tqdm')
    def test_predict_mask_numpy(self, mock_tqdm, mock_mirrored, mocked_membrain_seg, sample_3d_data):
        """Test predict_mask method with numpy input."""
        data = sample_3d_data['small']  # 8x8x8
        
        mocked_membrain_seg.preprocess = MagicMock(return_value=torch.randn(1, 1, 8, 8, 8))
        mocked_membrain_seg.inferer = MagicMock(return_value=torch.randn(1, 1, 8, 8, 8))
        mock_mirrored.side_effect = lambda x, m: x
        mock_tqdm.side_effect = lambda x, disable=False: x
        
        result = mocked_membrain_seg.predict_mask(data, threshold=0.5, test_time_augmentation=False)
        
        assert isinstance(result, np.ndarray)
        # Check that values are binary (0 or 1)
        unique_values = np.unique(result)
        assert all(val in [0, 1] for val in unique_values)
    
    @patch('torch_segment_membranes_3d.segment.get_mirrored_img')
    @patch('torch_segment_membranes_3d.segment.tqdm')
    def test_predict_with_tta(self, mock_tqdm, mock_mirrored, mocked_membrain_seg, sample_3d_data):
        """Test predict method with test time augmentation."""
        data = sample_3d_data['small']  # 8x8x8
        
        mocked_membrain_seg.preprocess = MagicMock(return_value=torch.randn(1, 1, 8, 8, 8))
        mocked_membrain_seg.inferer = MagicMock(return_value=torch.randn(1, 1, 8, 8, 8))
        mock_mirrored.side_effect = lambda x, m: x
        mock_tqdm.side_effect = lambda x, disable=False: x
        
        result = mocked_membrain_seg.predict_mask(data, test_time_augmentation=True)
        
        assert mocked_membrain_seg.inferer.call_count == 8  # TTA calls inferer 8 times
    
    @patch('torch_segment_membranes_3d.segment.get_mirrored_img')
    @patch('torch_segment_membranes_3d.segment.tqdm')
    def test_predict_threshold(self, mock_tqdm, mock_mirrored, mocked_membrain_seg):
        """Test predict method with threshold."""
        mock_pred = torch.tensor([[[[0.3, 0.7]]]])
        mocked_membrain_seg.preprocess = MagicMock(return_value=mock_pred)
        mocked_membrain_seg.inferer = MagicMock(return_value=mock_pred)
        mock_mirrored.side_effect = lambda x, m: x
        mock_tqdm.side_effect = lambda x, disable=False: x
        
        data = np.array([[[1.0]]])
        result = mocked_membrain_seg.predict_mask(data, threshold=0.5, test_time_augmentation=False)
        
        # Should apply threshold: 0.3 -> 0, 0.7 -> 1
        assert result.flatten()[0] == 0
        assert result.flatten()[1] == 1
    
    @patch('torch_segment_membranes_3d.segment.get_mirrored_img')
    @patch('torch_segment_membranes_3d.segment.tqdm')
    def test_predict_wrapper_method(self, mock_tqdm, mock_mirrored, mocked_membrain_seg, sample_3d_data):
        """Test that predict() method is a proper wrapper for predict_mask()."""
        data = sample_3d_data['medium']
        
        mocked_membrain_seg.preprocess = MagicMock(return_value=torch.randn(1, 1, 16, 16, 16))
        mocked_membrain_seg.inferer = MagicMock(return_value=torch.randn(1, 1, 16, 16, 16))
        mock_mirrored.side_effect = lambda x, m: x
        mock_tqdm.side_effect = lambda x, disable=False: x
        
        # Test that predict() and predict_mask() return the same result
        result_predict = mocked_membrain_seg.predict(data, threshold=0.5, test_time_augmentation=False)
        
        # Reset mocks
        mocked_membrain_seg.preprocess.reset_mock()
        mocked_membrain_seg.inferer.reset_mock()
        
        result_predict_mask = mocked_membrain_seg.predict_mask(data, threshold=0.5, test_time_augmentation=False)
        
        # Both should be numpy arrays and have same shape
        assert isinstance(result_predict, np.ndarray)
        assert isinstance(result_predict_mask, np.ndarray)
        assert result_predict.shape == result_predict_mask.shape
    
    def test_progress_bar_disabled(self, mocked_membrain_seg, sample_3d_data):
        """Test that progress bar can be disabled."""
        data = sample_3d_data['small']
        
        mocked_membrain_seg.preprocess = MagicMock(return_value=torch.randn(1, 1, 8, 8, 8))
        mocked_membrain_seg.inferer = MagicMock(return_value=torch.randn(1, 1, 8, 8, 8))
        
        with patch('torch_segment_membranes_3d.segment.get_mirrored_img') as mock_mirrored, \
             patch('torch_segment_membranes_3d.segment.tqdm') as mock_tqdm:
            
            mock_mirrored.side_effect = lambda x, m: x
            mock_tqdm.side_effect = lambda x, disable=False: x
            
            # Test with progress_bar=False
            mocked_membrain_seg.predict_probabilities(data, progress_bar=False, test_time_augmentation=False)
            
            # Check that tqdm was called with disable=True
            mock_tqdm.assert_called_with(range(1), disable=True)
    
    def test_model_target_shape_assignment(self, mock_checkpoint_path):
        """Test that model.target_shape is properly assigned."""
        with patch('torch_segment_membranes_3d.segment.utils.get_membrain_checkpoint') as mock_get_checkpoint, \
             patch('torch_segment_membranes_3d.segment.load_model_from_checkpoint') as mock_load, \
             patch('torch_segment_membranes_3d.segment.get_prediction_transforms'), \
             patch('torch_segment_membranes_3d.segment.SlidingWindowInferer'):
            
            mock_get_checkpoint.return_value = mock_checkpoint_path
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            
            seg = MembrainSeg(sw_window_size=128)
            
            # Check that target_shape was assigned to the model
            assert mock_model.target_shape == (128, 128, 128)


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================

class TestMembrainSegIntegration:
    """Integration-style tests for MembrainSeg (still mocked but more realistic)."""
    
    @patch('torch_segment_membranes_3d.segment.utils.get_membrain_checkpoint')
    @patch('torch_segment_membranes_3d.segment.load_model_from_checkpoint')
    @patch('torch_segment_membranes_3d.segment.get_prediction_transforms')
    @patch('torch_segment_membranes_3d.segment.SlidingWindowInferer')
    def test_full_prediction_pipeline(self, mock_inferer_class, mock_transforms, mock_load, mock_get_checkpoint, sample_3d_data):
        """Test the full prediction pipeline with more realistic mocking."""
        # Setup mocks
        mock_get_checkpoint.return_value = "/fake/checkpoint.ckpt"
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        # Mock transforms to return a tensor with expected shape
        def mock_transform(data):
            return torch.from_numpy(data).float()
        mock_transforms.return_value = mock_transform
        
        # Mock inferer instance
        mock_inferer_instance = MagicMock()
        mock_inferer_class.return_value = mock_inferer_instance
        
        # Create MembrainSeg instance
        seg = MembrainSeg()
        
        # Test data
        test_data = sample_3d_data['small']  # 8x8x8
        
        # Mock the inferer to return predictable output
        mock_output = torch.rand(1, 1, 8, 8, 8) * 0.6 + 0.2  # Values between 0.2 and 0.8
        mock_inferer_instance.return_value = mock_output
        
        with patch('torch_segment_membranes_3d.segment.get_mirrored_img') as mock_mirrored, \
             patch('torch_segment_membranes_3d.segment.tqdm') as mock_tqdm:
            
            mock_mirrored.side_effect = lambda x, m: x  # No actual mirroring
            mock_tqdm.side_effect = lambda x, disable=False: x  # Pass through range
            
            # Test probabilities
            probabilities = seg.predict_probabilities(test_data, test_time_augmentation=False)
            assert isinstance(probabilities, np.ndarray)
            assert probabilities.shape == test_data.shape
            
            # Test mask
            mask = seg.predict_mask(test_data, threshold=0.5, test_time_augmentation=False)
            assert isinstance(mask, np.ndarray)
            assert mask.shape == test_data.shape
            assert set(np.unique(mask)) <= {0, 1}  # Only binary values