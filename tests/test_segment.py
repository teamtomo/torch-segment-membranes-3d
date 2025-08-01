import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from torch_segment_membranes_3d.segment import MembrainSeg


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def sample_data():
    """Create sample 3D data for testing."""
    np.random.seed(42)
    return {
        'numpy': np.random.randn(32, 32, 32).astype(np.float32),
        'torch': torch.randn(32, 32, 32, dtype=torch.float32),
    }

@pytest.fixture
def mocked_membrain_seg():
    """Create a mocked MembrainSeg instance."""
    with patch('torch_segment_membranes_3d.segment.utils.get_membrain_checkpoint'), \
         patch('torch_segment_membranes_3d.segment.load_model_from_checkpoint') as mock_load, \
         patch('torch_segment_membranes_3d.segment.get_prediction_transforms') as mock_transforms, \
         patch('torch_segment_membranes_3d.segment.SlidingWindowInferer') as mock_inferer_class:
        
        # Setup mocks
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        mock_transforms.return_value = lambda x: torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x
        
        mock_inferer = MagicMock()
        mock_inferer.return_value = torch.rand(1, 1, 32, 32, 32) * 0.8 + 0.1  # Values 0.1-0.9
        mock_inferer_class.return_value = mock_inferer
        
        seg = MembrainSeg()
        seg.inferer = mock_inferer
        return seg


# =============================================================================
# TESTS
# =============================================================================

class TestMembrainSeg:
    """Test MembrainSeg functionality."""
    
    @patch('torch_segment_membranes_3d.segment.utils.get_membrain_checkpoint')
    @patch('torch_segment_membranes_3d.segment.load_model_from_checkpoint')
    @patch('torch_segment_membranes_3d.segment.get_prediction_transforms')
    @patch('torch_segment_membranes_3d.segment.SlidingWindowInferer')
    def test_init_default(self, mock_inferer, mock_transforms, mock_load, mock_checkpoint):
        """Test default initialization."""
        mock_checkpoint.return_value = "/fake/checkpoint.ckpt"
        mock_load.return_value = MagicMock()
        
        seg = MembrainSeg()
        
        assert seg.sw_batch_size == 4
        assert seg.sw_window_size == 160
    
    @patch('torch_segment_membranes_3d.segment.utils.get_membrain_checkpoint')
    @patch('torch_segment_membranes_3d.segment.load_model_from_checkpoint')
    @patch('torch_segment_membranes_3d.segment.get_prediction_transforms')
    @patch('torch_segment_membranes_3d.segment.SlidingWindowInferer')
    def test_init_custom_params(self, mock_inferer, mock_transforms, mock_load, mock_checkpoint):
        """Test initialization with custom parameters."""
        mock_checkpoint.return_value = "/fake/checkpoint.ckpt"
        mock_load.return_value = MagicMock()
        
        seg = MembrainSeg(sw_batch_size=8, sw_window_size=256)
        
        assert seg.sw_batch_size == 8
        assert seg.sw_window_size == 256
    
    def test_preprocess_numpy(self, mocked_membrain_seg, sample_data):
        """Test preprocessing with numpy input."""
        result = mocked_membrain_seg.preprocess(sample_data['numpy'])
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1  # batch dimension
    
    def test_preprocess_torch(self, mocked_membrain_seg, sample_data):
        """Test preprocessing with torch input."""
        result = mocked_membrain_seg.preprocess(sample_data['torch'])
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1  # batch dimension
    
    @patch('torch_segment_membranes_3d.segment.get_mirrored_img')
    @patch('torch_segment_membranes_3d.segment.tqdm')
    def test_predict_probabilities_numpy(self, mock_tqdm, mock_mirror, mocked_membrain_seg, sample_data):
        """Test predict_probabilities with numpy input."""
        mock_mirror.side_effect = lambda x, m: x
        mock_tqdm.side_effect = lambda x, disable=False: x
        
        result = mocked_membrain_seg.predict_probabilities(sample_data['numpy'], test_time_augmentation=False)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_data['numpy'].shape
    
    @patch('torch_segment_membranes_3d.segment.get_mirrored_img')
    @patch('torch_segment_membranes_3d.segment.tqdm')
    def test_predict_probabilities_torch(self, mock_tqdm, mock_mirror, mocked_membrain_seg, sample_data):
        """Test predict_probabilities with torch input."""
        mock_mirror.side_effect = lambda x, m: x
        mock_tqdm.side_effect = lambda x, disable=False: x
        
        result = mocked_membrain_seg.predict_probabilities(sample_data['torch'], test_time_augmentation=False)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == sample_data['torch'].shape
    
    @patch('torch_segment_membranes_3d.segment.get_mirrored_img')
    @patch('torch_segment_membranes_3d.segment.tqdm')
    def test_predict_mask(self, mock_tqdm, mock_mirror, mocked_membrain_seg, sample_data):
        """Test predict_mask produces binary output."""
        mock_mirror.side_effect = lambda x, m: x
        mock_tqdm.side_effect = lambda x, disable=False: x
        
        result = mocked_membrain_seg.predict_mask(sample_data['numpy'], threshold=0.5, test_time_augmentation=False)
        
        assert isinstance(result, np.ndarray)
        assert set(np.unique(result)) <= {0, 1}  # Binary output
    
    @patch('torch_segment_membranes_3d.segment.get_mirrored_img')
    @patch('torch_segment_membranes_3d.segment.tqdm')
    def test_predict_wrapper(self, mock_tqdm, mock_mirror, mocked_membrain_seg, sample_data):
        """Test predict() wrapper function."""
        mock_mirror.side_effect = lambda x, m: x
        mock_tqdm.side_effect = lambda x, disable=False: x
        
        result = mocked_membrain_seg.predict(sample_data['numpy'], test_time_augmentation=False)
        
        assert isinstance(result, np.ndarray)
        assert set(np.unique(result)) <= {0, 1}  # Binary output
    
    @patch('torch_segment_membranes_3d.segment.get_mirrored_img')
    @patch('torch_segment_membranes_3d.segment.tqdm')
    def test_test_time_augmentation(self, mock_tqdm, mock_mirror, mocked_membrain_seg, sample_data):
        """Test test-time augmentation calls inferer 8 times."""
        mock_mirror.side_effect = lambda x, m: x
        mock_tqdm.side_effect = lambda x, disable=False: x
        
        mocked_membrain_seg.inferer.reset_mock()
        mocked_membrain_seg.predict_probabilities(sample_data['numpy'], test_time_augmentation=True, progress_bar=False)
        
        assert mocked_membrain_seg.inferer.call_count == 8