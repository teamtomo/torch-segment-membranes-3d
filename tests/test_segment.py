import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from torch_segment_membranes_3d.segment import MembrainSeg


class TestMembrainSeg:
    """Minimal tests for MembrainSeg to achieve 70% coverage."""
    
    @patch('torch_segment_membranes_3d.segment.utils.get_membrain_checkpoint')
    @patch('torch_segment_membranes_3d.segment.PreprocessedSemanticSegmentationUnet.load_from_checkpoint')
    @patch('torch_segment_membranes_3d.segment.get_prediction_transforms')
    @patch('torch_segment_membranes_3d.segment.SlidingWindowInferer')
    def test_init(self, mock_inferer, mock_transforms, mock_load_checkpoint, mock_get_checkpoint):
        """Test MembrainSeg initialization."""
        mock_get_checkpoint.return_value = "/fake/checkpoint.ckpt"
        mock_model = MagicMock()
        mock_load_checkpoint.return_value = mock_model
        mock_transforms.return_value = MagicMock()
        
        seg = MembrainSeg()
        
        assert seg.sw_batch_size == 4
        assert seg.sw_window_size == 160
        mock_model.eval.assert_called_once()
    
    def setup_seg(self):
        """Helper to create mocked MembrainSeg."""
        with patch('torch_segment_membranes_3d.segment.utils.get_membrain_checkpoint'), \
             patch('torch_segment_membranes_3d.segment.PreprocessedSemanticSegmentationUnet.load_from_checkpoint'), \
             patch('torch_segment_membranes_3d.segment.get_prediction_transforms'), \
             patch('torch_segment_membranes_3d.segment.SlidingWindowInferer'):
            return MembrainSeg()
    
    def test_preprocess_numpy(self):
        """Test preprocessing numpy array."""
        seg = self.setup_seg()
        seg.transforms = MagicMock(return_value=torch.randn(1, 32, 32, 32))
        
        data = np.random.randn(32, 32, 32)
        result = seg.preprocess(data)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1  # batch dimension
    
    def test_preprocess_torch(self):
        """Test preprocessing torch tensor."""
        seg = self.setup_seg()
        seg.transforms = MagicMock(return_value=torch.randn(1, 16, 16, 16))
        
        data = torch.randn(16, 16, 16)
        result = seg.preprocess(data)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 1
    
    def test_preprocess_no_normalization(self):
        """Test preprocessing without normalization."""
        seg = self.setup_seg()
        seg.transforms = MagicMock(return_value=torch.randn(1, 8, 8, 8))
        
        data = np.random.randn(8, 8, 8)
        result = seg.preprocess(data, normalize_data=False)
        
        assert isinstance(result, torch.Tensor)
    
    @patch('torch_segment_membranes_3d.segment.get_mirrored_img')
    @patch('torch_segment_membranes_3d.segment.tqdm')
    def test_run_numpy(self, mock_tqdm, mock_mirrored):
        """Test run method with numpy input."""
        seg = self.setup_seg()
        seg.preprocess = MagicMock(return_value=torch.randn(1, 1, 16, 16, 16))
        seg.inferer = MagicMock(return_value=torch.randn(1, 1, 16, 16, 16))
        mock_mirrored.side_effect = lambda x, m: x
        mock_tqdm.side_effect = lambda x, disable=False: x
        
        data = np.random.randn(16, 16, 16)
        result = seg.run(data, test_time_augmentation=False)
        
        assert isinstance(result, np.ndarray)
    
    @patch('torch_segment_membranes_3d.segment.get_mirrored_img')
    @patch('torch_segment_membranes_3d.segment.tqdm')
    def test_run_torch(self, mock_tqdm, mock_mirrored):
        """Test run method with torch input."""
        seg = self.setup_seg()
        seg.preprocess = MagicMock(return_value=torch.randn(1, 1, 8, 8, 8))
        seg.inferer = MagicMock(return_value=torch.randn(1, 1, 8, 8, 8))
        mock_mirrored.side_effect = lambda x, m: x
        mock_tqdm.side_effect = lambda x, disable=False: x
        
        data = torch.randn(8, 8, 8)
        result = seg.run(data, test_time_augmentation=False)
        
        assert isinstance(result, torch.Tensor)
    
    @patch('torch_segment_membranes_3d.segment.get_mirrored_img')
    @patch('torch_segment_membranes_3d.segment.tqdm')
    def test_run_with_tta(self, mock_tqdm, mock_mirrored):
        """Test run method with test time augmentation."""
        seg = self.setup_seg()
        seg.preprocess = MagicMock(return_value=torch.randn(1, 1, 4, 4, 4))
        seg.inferer = MagicMock(return_value=torch.randn(1, 1, 4, 4, 4))
        mock_mirrored.side_effect = lambda x, m: x
        mock_tqdm.side_effect = lambda x, disable=False: x
        
        data = np.random.randn(4, 4, 4)
        result = seg.run(data, test_time_augmentation=True)
        
        assert seg.inferer.call_count == 8  # TTA calls inferer 8 times
    
    @patch('torch_segment_membranes_3d.segment.get_mirrored_img')
    @patch('torch_segment_membranes_3d.segment.tqdm')
    def test_run_threshold(self, mock_tqdm, mock_mirrored):
        """Test run method with threshold."""
        seg = self.setup_seg()
        mock_pred = torch.tensor([[[[0.3, 0.7]]]])
        seg.preprocess = MagicMock(return_value=mock_pred)
        seg.inferer = MagicMock(return_value=mock_pred)
        mock_mirrored.side_effect = lambda x, m: x
        mock_tqdm.side_effect = lambda x, disable=False: x
        
        data = np.array([[1.0]])
        result = seg.run(data, threshold=0.5, test_time_augmentation=False)
        
        # Should apply threshold: 0.3 -> 0, 0.7 -> 1
        assert result[0] == 0
        assert result[1] == 1