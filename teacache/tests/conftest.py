"""Shared fixtures. The first statements make the whole session offline before anything imports
mflux or huggingface_hub (huggingface_hub reads HF_HUB_OFFLINE and its cache paths once, at import): no
test may download a model or read the user's model cache."""

import atexit
import os
import shutil
import tempfile

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = tempfile.mkdtemp(prefix="mflux-teacache-hf-home-")
# Each of these overrides HF_HOME when the user has set it, so point both into the temporary home too.
os.environ["HF_HUB_CACHE"] = os.path.join(os.environ["HF_HOME"], "hub")
os.environ["HF_XET_CACHE"] = os.path.join(os.environ["HF_HOME"], "xet")
atexit.register(shutil.rmtree, os.environ["HF_HOME"], ignore_errors=True)

import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import huggingface_hub.constants
import pytest
from _fakes import FakeQwenImage, FakeTeaCache, FakeZImage, LoopingFakeQwenImage
from mflux.models.qwen.cli import qwen_image_generate
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.z_image.cli import z_image_generate
from mflux.models.z_image.variants.z_image import ZImage
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil

from mflux.extras.teacache import _core, z_image

# The constant is what huggingface_hub's request session checks; force it even if something imported the hub first.
huggingface_hub.constants.HF_HUB_OFFLINE = True
assert huggingface_hub.constants.is_offline_mode(), "the test session must run with the Hugging Face Hub offline"

# One entry per command: the real model class, which no test may build, and the mflux command module whose
# model-class name the run_command fixture points at the fake (attribute name, fake class). Each fake class keeps its
# own instances and fail_on_seed. tests/test_isolation.py checks that every console script's adapter is covered here.
REAL_PIPELINES: tuple[type, ...] = (ZImage, QwenImage)
FAKES: dict[ModuleType, tuple[str, type]] = {
    z_image_generate: ("ZImage", FakeZImage),
    qwen_image_generate: ("QwenImage", FakeQwenImage),
}


def _no_real_model(self: Any, *args: Any, **kwargs: Any) -> None:
    raise AssertionError(
        f"a test constructed the real {type(self).__name__}, which would load weights; run the command through "
        "the run_command fixture (its fake) instead"
    )


def _clear_fakes() -> None:
    for _, fake in FAKES.values():
        fake.instances.clear()
        fake.fail_on_seed.clear()
    # The looping Qwen fake shares FakeQwenImage's lists; its weak reference to the last transformer and its decode
    # hook are its own.
    LoopingFakeQwenImage.transformer_ref = None
    LoopingFakeQwenImage.decode_hook = None


@pytest.fixture(autouse=True)
def isolated_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    # Child processes (the console-script test) inherit these; this process read them above.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf-home" / "hub"))
    monkeypatch.setenv("HF_XET_CACHE", str(tmp_path / "hf-home" / "xet"))
    # A default --output (image.png) or any stray file lands in the test's own directory, never the repo.
    monkeypatch.chdir(tmp_path)
    # Each fake read the real signatures at import; the real classes themselves can no longer be built.
    for pipeline in REAL_PIPELINES:
        monkeypatch.setattr(pipeline, "__init__", _no_real_model)
    _clear_fakes()
    # parse_args sets GeneratedImage.model_path and can switch off EXIF embedding process-wide.
    monkeypatch.setattr(GeneratedImage, "model_path", None)
    monkeypatch.setattr(ImageUtil, "embed_metadata_enabled", True)
    yield
    _clear_fakes()


@pytest.fixture
def teacache(monkeypatch: pytest.MonkeyPatch) -> FakeTeaCache:
    fake = FakeTeaCache()
    monkeypatch.setattr(_core, "apply_teacache", fake)
    return fake


@pytest.fixture
def run_command(monkeypatch: pytest.MonkeyPatch, teacache: FakeTeaCache) -> Callable[..., None]:
    # Only tests that run a command get the fake model classes. It is not autouse on purpose: on
    # Python 3.14 annotations are evaluated lazily from module globals, so a global model-class swap would
    # also change what the drift test reads as a command's load() return annotation.
    for module, (name, fake) in FAKES.items():
        monkeypatch.setattr(module, name, fake)

    def _run(argv: list[str], adapter: _core.Adapter | None = None) -> None:
        adapter = adapter or z_image.ADAPTER
        monkeypatch.setattr(sys, "argv", [adapter.command, *argv])
        _core.run(adapter)

    return _run
