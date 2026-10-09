"""The mflux PR #776 namespace contract as this distribution keeps it."""

import subprocess
import sys
from pathlib import Path

import pytest

import mflux

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CORE_PARENT = Path(mflux.__file__).resolve().parents[1]


def test_source_tree_has_no_parent_initializers() -> None:
    """Bug: someone adds src/mflux/__init__.py or src/mflux/extras/__init__.py; the editable
    install then puts that file in front of mflux core's and `import mflux` loses core."""
    assert not (SRC / "mflux" / "__init__.py").exists()
    assert not (SRC / "mflux" / "extras" / "__init__.py").exists()
    assert (SRC / "mflux" / "extras" / "myplugin" / "__init__.py").is_file()


@pytest.mark.parametrize("core_first", [True, False])
def test_imports_next_to_core_and_another_extras_package(tmp_path: Path, core_first: bool) -> None:
    """Bug: a parent __init__.py in this tree turns mflux.extras into a regular package that hides
    every other mflux.extras.* distribution, or this package's import pulls in mflux.models."""
    other_root = tmp_path / "other_distribution"
    other = other_root / "mflux" / "extras" / "other"
    other.mkdir(parents=True)
    (other / "__init__.py").write_text("VALUE = 7\n", encoding="utf-8")
    paths = [CORE_PARENT, SRC, other_root] if core_first else [other_root, SRC, CORE_PARENT]
    # mflux core's extend_path keeps core's own directory first, then follows sys.path; the first one that carries
    # this package wins. In the dev environment that is src/ (the editable install adds it through a .pth file); where
    # the built wheel is installed, core's directory carries the installed copy.
    search_order = [CORE_PARENT, *(path for path in paths if path != CORE_PARENT)]
    myplugin_init = next(
        init for path in search_order if (init := path / "mflux" / "extras" / "myplugin" / "__init__.py").is_file()
    )
    code = (
        f"import sys; sys.path[:0] = {[str(path) for path in paths]!r}\n"
        "import mflux, mflux.extras.myplugin, mflux.extras.other\n"
        f"assert mflux.__file__ == {str(CORE_PARENT / 'mflux' / '__init__.py')!r}, mflux.__file__\n"
        "assert mflux.extras.__file__ is None\n"
        "assert mflux.extras.other.VALUE == 7\n"
        f"assert mflux.extras.myplugin.__file__ == {str(myplugin_init)!r}\n"
        "assert 'mflux.models' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
