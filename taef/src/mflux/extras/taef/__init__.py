"""Adds commands that run mflux's Z-Image generation and write a live preview image while it denoises, decoded by
mlx-taef. The package needs mflux and mlx-taef; installing it brings in both."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mflux-taef")
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0+unknown"
