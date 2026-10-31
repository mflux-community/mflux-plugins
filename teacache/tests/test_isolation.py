"""The guards that keep every test offline, away from real weights, and out of the repository."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import huggingface_hub.constants
import pytest
from conftest import FAKES, REAL_PIPELINES

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

TESTS = Path(__file__).resolve().parent


def test_the_hub_is_offline_and_its_cache_is_a_private_empty_directory() -> None:
    """Bug: HF_HUB_OFFLINE / HF_HOME are set after huggingface_hub was imported (it reads them once), so a stray Hub
    call downloads a model or a test reads the user's real model cache."""
    assert huggingface_hub.constants.is_offline_mode()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    hub_cache = Path(huggingface_hub.constants.HF_HUB_CACHE)
    assert hub_cache.parent.name.startswith("mflux-teacache-hf-home-")
    assert not hub_cache.exists() or not any(hub_cache.iterdir())


def test_the_users_own_cache_variables_are_replaced(tmp_path: Path) -> None:
    """Bug: conftest sets only HF_HOME, so a user who sets HF_HUB_CACHE or HF_XET_CACHE (each overrides HF_HOME) still
    runs the tests against their real model cache."""
    real = tmp_path / "users-cache"
    env = {**os.environ, "HF_HUB_CACHE": str(real / "hub"), "HF_XET_CACHE": str(real / "xet")}
    code = "import conftest, huggingface_hub.constants as c; print(c.HF_HOME); print(c.HF_HUB_CACHE); print(c.HF_XET_CACHE)"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=TESTS, env=env, capture_output=True, text=True, check=False, timeout=120
    )
    assert result.returncode == 0, result.stderr
    home, hub, xet = (Path(line) for line in result.stdout.splitlines())
    assert home.name.startswith("mflux-teacache-hf-home-")
    assert (hub.parent, xet.parent) == (home, home)


@pytest.mark.parametrize("pipeline", REAL_PIPELINES, ids=lambda pipeline: pipeline.__name__)
def test_the_real_model_class_cannot_be_built(pipeline: type) -> None:
    """Bug: the autouse guard is dropped for one model class, so a test that skips the run_command fixture constructs
    the real class and loads weights from a cache or the Hub."""
    with pytest.raises(AssertionError, match=f"constructed the real {pipeline.__name__}"):
        pipeline()


def test_every_command_has_a_guard_and_a_fake() -> None:
    """Bug: a command's model class is missing from REAL_PIPELINES or its mflux command module from FAKES, so a test
    of that command builds the real class (and loads weights) instead of the fake."""
    scripts = tomllib.loads((TESTS.parent / "pyproject.toml").read_text())["project"]["scripts"]
    adapters = [importlib.import_module(target.split(":")[0]).ADAPTER for target in scripts.values()]
    assert {adapter.pipeline for adapter in adapters} == set(REAL_PIPELINES)
    assert {sys.modules[adapter.load.__module__] for adapter in adapters} == set(FAKES)
    for module, (name, fake) in FAKES.items():
        assert getattr(module, name) is fake.PIPELINE, f"{fake.__name__} replaces {module.__name__}.{name}"


def test_each_test_runs_in_its_own_directory(tmp_path: Path) -> None:
    """Bug: the working directory is not isolated, so a run without --output writes image.png into the repository."""
    assert Path.cwd().resolve() == tmp_path.resolve()
