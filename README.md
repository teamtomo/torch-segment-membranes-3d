# torch-segment-membranes-3d

[![License](https://img.shields.io/pypi/l/torch-segment-membranes-3d.svg?color=green)](https://github.com/teamtomo/torch-segment-membranes-3d/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-segment-membranes-3d.svg?color=green)](https://pypi.org/project/torch-segment-membranes-3d)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-segment-membranes-3d.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/torch-segment-membranes-3d/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/torch-segment-membranes-3d/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/teamtomo/torch-segment-membranes-3d/branch/main/graph/badge.svg)](https://codecov.io/gh/teamtomo/torch-segment-membranes-3d)

Semantic segmentation of membranes in 3D cryo-ET volumes, this is a minimal re-implementation of the original membrain-seg repository purely for inference. To retrain or fine-tune public weights, refer to the original github repository.

## Quick Start

```
from torch_segment_membranes_3d import segment

# Initialize the Segmenter Class
segmenter = segment.MembrainSeg()

# Run 
segmentation = segmenter.run(data)
```

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork teamtomo/torch-segment-membranes-3d --clone
# or just
# gh repo clone teamtomo/torch-segment-membranes-3d
cd torch-segment-membranes-3d
uv sync
```

Run tests:

```sh
uv run pytest
```

Lint files:

```sh
uv run pre-commit run --all-files
```
