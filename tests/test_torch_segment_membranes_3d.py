import torch_segment_membranes_3d


def test_imports_with_version():
    assert isinstance(torch_segment_membranes_3d.__version__, str)
