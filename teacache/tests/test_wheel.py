"""What a built wheel actually carries."""

import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> zipfile.ZipFile:
    uv = shutil.which("uv")
    assert uv is not None, "uv must be on PATH to build the wheel"
    out = tmp_path_factory.mktemp("wheel")
    result = subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(out), str(ROOT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, f"uv build failed (exit {result.returncode}):\n{result.stderr}"
    (path,) = out.glob("*.whl")
    return zipfile.ZipFile(path)


def dist_info(wheel: zipfile.ZipFile) -> str:
    """The wheel's .dist-info directory, from its own file name (name-version-tags.whl), so a version bump or a
    renamed distribution needs no edit here."""
    name, version = Path(wheel.filename).name.split("-")[:2]
    return f"{name}-{version}.dist-info"


def test_wheel_ships_only_the_teacache_child(wheel: zipfile.ZipFile) -> None:
    """Bug: [tool.uv.build-backend] module-name widened to "mflux" (or namespace dropped), so the
    wheel carries mflux/__init__.py and overwrites mflux core's on install."""
    names = wheel.namelist()
    assert "mflux/__init__.py" not in names
    assert "mflux/extras/__init__.py" not in names
    package_files = [name for name in names if not name.startswith(dist_info(wheel))]
    assert [n for n in package_files if n not in ("mflux/", "mflux/extras/")] != []
    assert all(
        name in ("mflux/", "mflux/extras/") or name.startswith("mflux/extras/teacache/") for name in package_files
    ), package_files


def requirements(wheel: zipfile.ZipFile) -> list[Requirement]:
    return [
        Requirement(line.split(":", 1)[1].strip())
        for line in wheel.read(f"{dist_info(wheel)}/METADATA").decode().splitlines()
        if line.startswith("Requires-Dist:")
    ]


def locked_version(name: str) -> Version:
    """The highest version uv.lock holds for one package (uv may lock several, one per marker). A regex, not tomllib,
    because the tests also run on Python 3.10."""
    lock = (ROOT / "uv.lock").read_text()
    versions = re.findall(rf'\[\[package\]\]\nname = "{re.escape(name)}"\nversion = "([^"]+)"', lock)
    assert versions, f"{name} is not in uv.lock"
    return max(Version(v) for v in versions)


def test_wheel_declares_the_command_and_no_mflux_upper_bound(wheel: zipfile.ZipFile) -> None:
    """Bug: the console script points at another module or loses the mflux-generate prefix; the mlx-teacache
    requirement loses its [mflux] extra or its 0.14.0 floor; or an mflux requirement gains an upper bound, so a newer
    mflux is refused at install instead of running with a warning."""
    entry_points = wheel.read(f"{dist_info(wheel)}/entry_points.txt").decode()
    assert "mflux-generate-z-image-teacache = mflux.extras.teacache.z_image:main" in entry_points
    requires = requirements(wheel)
    (teacache,) = [r for r in requires if r.name == "mlx-teacache"]
    assert "mflux" in teacache.extras and str(teacache.specifier) == ">=0.14.0"
    (mflux,) = [r for r in requires if r.name == "mflux"]
    assert str(mflux.specifier) == ">=0.22"
    bounded = [
        str(r) for r in requires if r.name in ("mflux", "mlx-teacache") for spec in r.specifier if "<" in spec.operator
    ]
    assert bounded == []


def test_wheel_keeps_pillow_below_13(wheel: zipfile.ZipFile) -> None:
    """Bug: the pillow requirement is dropped, or a marker narrows it (say to Python 3.10 only), so a fresh install on
    Python 3.11 or newer picks pillow 13, which removed PIL._typing, and `import mflux` fails in every mflux up to
    0.23.0 before the command starts."""
    pillows = [r for r in requirements(wheel) if r.name == "pillow"]
    assert len(pillows) == 1, f"expected one pillow requirement, got {[str(r) for r in pillows]}"
    (pillow,) = pillows
    assert str(pillow.specifier) == "<13"
    assert pillow.marker is None, f"{pillow} applies to some Pythons only"


def test_pillow_cap_is_reviewed_once_the_lock_moves_past_mflux_0_23_0(wheel: zipfile.ZipFile) -> None:
    """Bug: someone upgrades the lock to an mflux newer than 0.23.0, which may carry mflux PR 890 (the Pillow 13 import
    fix), and pillow<13 stays without anyone checking, so users are held on pillow 12 and its open advisories. Nothing
    moves the lock by itself, so this fires on the next manual lock upgrade, not on the day mflux ships."""
    capped = any(r.name == "pillow" for r in requirements(wheel))
    mflux = locked_version("mflux")
    assert not (capped and mflux > Version("0.23.0")), (
        f"uv.lock now holds mflux {mflux}. If that release carries mflux PR 890 "
        "(https://github.com/mflux-community/mflux/pull/890), drop pillow<13 from teacache/pyproject.toml and "
        "template/pyproject.toml, delete the pillow tests here and in template/tests/test_wheel.py, and release."
    )


def test_wheel_carries_the_licence_and_the_mflux_notice(wheel: zipfile.ZipFile) -> None:
    """Bug: license-files drops NOTICE (or the build falls back to license = { file = "LICENSE" }), so the wheel
    ships statements copied from mflux's MIT-licensed main() without mflux's copyright notice."""
    names = wheel.namelist()
    assert f"{dist_info(wheel)}/licenses/LICENSE" in names
    assert f"{dist_info(wheel)}/licenses/NOTICE" in names
    notice = wheel.read(f"{dist_info(wheel)}/licenses/NOTICE").decode()
    assert "MIT License" in notice and "Filip Strand" in notice
    metadata = wheel.read(f"{dist_info(wheel)}/METADATA").decode().splitlines()
    assert "License-Expression: Apache-2.0" in metadata


def test_wheel_claims_only_the_python_versions_it_is_tested_on(wheel: zipfile.ZipFile) -> None:
    """Bug: a "Programming Language :: Python :: 3.X" classifier names a version outside 3.10 to 3.13 (the range CI
    tests, at both ends), so PyPI advertises support nobody checked."""
    metadata = wheel.read(f"{dist_info(wheel)}/METADATA").decode().splitlines()
    prefix = "Classifier: Programming Language :: Python :: 3."
    minors = sorted(int(line.removeprefix(prefix)) for line in metadata if line.startswith(prefix))
    assert minors == [10, 11, 12, 13]
