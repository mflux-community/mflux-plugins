"""TeaCache step skipping for mflux's command line, powered by mlx-teacache."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mflux-teacache")
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0+unknown"
