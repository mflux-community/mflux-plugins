"""The memory notice points at a README section by name; that section must exist under that name."""

from argparse import Namespace
from pathlib import Path

from mflux.extras.taef import _core


def test_the_readme_has_the_heading_the_notice_points_to() -> None:
    """Bug: the README section is renamed (or dropped), or the notice quotes another name, so the note points at
    nothing."""
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert "## Memory at 1024×1024" in readme.splitlines()
    args = Namespace(low_ram=False, mlx_cache_limit_gb=None, width=1024, height=1024, image_path=None)
    printed = _core.memory_notice("cmd", args)
    assert printed is not None and '(see "Memory at 1024×1024" in the mflux-taef README)' in printed
