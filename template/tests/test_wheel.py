"""What a built wheel actually carries."""

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> zipfile.ZipFile:
    uv = shutil.which("uv")
    assert uv is not None, "uv must be on PATH to build the wheel"
    # This stays offline: uv's version is inside the uv_build range in pyproject.toml (CI pins it), so uv builds with
    # its own backend and downloads nothing.
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


def test_wheel_ships_only_the_plugin_child(wheel: zipfile.ZipFile) -> None:
    """Bug: [tool.uv.build-backend] module-name widened to "mflux" (or namespace dropped), so the
    wheel carries mflux/__init__.py and overwrites mflux core's on install."""
    names = wheel.namelist()
    assert "mflux/__init__.py" not in names
    assert "mflux/extras/__init__.py" not in names
    package_files = [name for name in names if not name.startswith(dist_info(wheel))]
    assert [n for n in package_files if n not in ("mflux/", "mflux/extras/")] != []
    assert all(
        name in ("mflux/", "mflux/extras/") or name.startswith("mflux/extras/myplugin/") for name in package_files
    ), package_files


def test_wheel_declares_the_command_and_an_mflux_floor_without_a_cap(wheel: zipfile.ZipFile) -> None:
    """Bug: the console script points at another module, or the mflux requirement loses its floor or gains an upper
    bound, so the plugin installs on an mflux without its steps or silently downgrades a newer one."""
    entry_points = wheel.read(f"{dist_info(wheel)}/entry_points.txt").decode()
    assert "mflux-generate-z-image-myplugin = mflux.extras.myplugin.z_image:main" in entry_points
    requires = [
        Requirement(line.split(":", 1)[1].strip())
        for line in wheel.read(f"{dist_info(wheel)}/METADATA").decode().splitlines()
        if line.startswith("Requires-Dist:")
    ]
    (mflux,) = [r for r in requires if r.name == "mflux"]
    operators = {specifier.operator for specifier in mflux.specifier}
    assert operators & {">=", ">"}, f"{mflux} has no lower bound"
    assert not operators & {"<", "<=", "==", "===", "~="}, f"{mflux} has an upper bound or a pin"


def test_wheel_carries_the_license_and_the_mflux_notice(wheel: zipfile.ZipFile) -> None:
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


def test_the_template_wheel_can_never_be_uploaded(wheel: zipfile.ZipFile) -> None:
    """Bug: the Private classifier is dropped, so a mistaken release would publish mflux-myplugin to PyPI.
    Delete this test when you rename the template."""
    metadata = wheel.read(f"{dist_info(wheel)}/METADATA").decode().splitlines()
    assert "Classifier: Private :: Do Not Upload" in metadata
