"""Template for a new mflux plugin. Rename myplugin to your plugin's name; see the repository's CONTRIBUTING.md."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mflux-myplugin")
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0+unknown"
