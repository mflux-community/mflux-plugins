"""Shared fixtures. The first statements make the whole session offline before anything imports
mflux or huggingface_hub (huggingface_hub reads HF_HUB_OFFLINE and HF_HOME once, at import): no test
may download a model or read the user's model cache."""

import atexit
import os
import shutil
import tempfile

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = tempfile.mkdtemp(prefix="mflux-teacache-hf-home-")
atexit.register(shutil.rmtree, os.environ["HF_HOME"], ignore_errors=True)

import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import huggingface_hub.constants
import pytest
from _fakes import FakeTeaCache, FakeZImage
from mflux.models.z_image.cli import z_image_generate
from mflux.models.z_image.variants.z_image import ZImage
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil

from mflux.extras.teacache import _core, z_image

# The constant is what huggingface_hub's request session checks; force it even if something imported the hub first.
huggingface_hub.constants.HF_HUB_OFFLINE = True
assert huggingface_hub.constants.is_offline_mode(), "the test session must run with the Hugging Face Hub offline"

COMMAND = "mflux-generate-z-image-teacache"


def _no_real_model(self: Any, *args: Any, **kwargs: Any) -> None:
    raise AssertionError(
        "a test constructed the real ZImage, which would load weights; run the command through the "
        "run_command fixture (FakeZImage) instead"
    )


@pytest.fixture(autouse=True)
def isolated_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    # Child processes (the console-script test) inherit these; this process read them above.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    # A default --output (image.png) or any stray file lands in the test's own directory, never the repo.
    monkeypatch.chdir(tmp_path)
    # FakeZImage read the real signature at import; the real class itself can no longer be built.
    monkeypatch.setattr(ZImage, "__init__", _no_real_model)
    FakeZImage.instances.clear()
    FakeZImage.fail_on_seed.clear()
    # parse_args sets GeneratedImage.model_path and can switch off EXIF embedding process-wide.
    monkeypatch.setattr(GeneratedImage, "model_path", None)
    monkeypatch.setattr(ImageUtil, "embed_metadata_enabled", True)
    yield
    FakeZImage.instances.clear()
    FakeZImage.fail_on_seed.clear()


@pytest.fixture
def teacache(monkeypatch: pytest.MonkeyPatch) -> FakeTeaCache:
    fake = FakeTeaCache()
    monkeypatch.setattr(_core, "apply_teacache", fake)
    return fake


@pytest.fixture
def run_command(monkeypatch: pytest.MonkeyPatch, teacache: FakeTeaCache) -> Callable[..., None]:
    # Only tests that run the command get the fake model class. It is not autouse on purpose: on
    # Python 3.14 annotations are evaluated lazily from module globals, so a global ZImage swap would
    # also change what the drift test reads as ZImageCommand.load()'s return annotation.
    monkeypatch.setattr(z_image_generate, "ZImage", FakeZImage)

    def _run(argv: list[str], adapter: _core.Adapter | None = None) -> None:
        monkeypatch.setattr(sys, "argv", [COMMAND, *argv])
        _core.run(adapter or z_image.ADAPTER)

    return _run
