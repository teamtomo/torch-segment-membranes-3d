"""Semantic segmentation of membranes in 3D cryo-ET volumes"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-segment-membranes-3d")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "TeamTomo"
__email__ = "alisterburt@gmail.com"
