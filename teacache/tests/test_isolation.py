"""The guards that keep every test offline, away from real weights, and out of the repository."""

import os
from pathlib import Path

import huggingface_hub.constants
import pytest
from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.z_image.variants.z_image import ZImage


def test_the_hub_is_offline_and_its_cache_is_a_private_empty_directory() -> None:
    """Bug: HF_HUB_OFFLINE / HF_HOME are set after huggingface_hub was imported (it reads them once), so a stray Hub
    call downloads a model or a test reads the user's real model cache."""
    assert huggingface_hub.constants.is_offline_mode()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    hub_cache = Path(huggingface_hub.constants.HF_HUB_CACHE)
    assert hub_cache.parent.name.startswith("mflux-teacache-hf-home-")
    assert not hub_cache.exists() or not any(hub_cache.iterdir())


def test_the_real_model_class_cannot_be_built() -> None:
    """Bug: the autouse guard is dropped, so a test that skips the run_command fixture constructs the real ZImage and
    loads weights from a cache or the Hub."""
    with pytest.raises(AssertionError, match="constructed the real ZImage"):
        ZImage(model_config=AVAILABLE_MODELS["z-image"])


def test_each_test_runs_in_its_own_directory(tmp_path: Path) -> None:
    """Bug: the working directory is not isolated, so a run without --output writes image.png into the repository."""
    assert Path.cwd().resolve() == tmp_path.resolve()
