"""The real mlx-taef preview callback, run through mflux's real callback dispatcher. The model is LoopingFakeZImage,
which runs mflux's denoising loop shape on noise latents, and the decoder is a randomly initialised mlx-taef Z-Image
decoder written to the test's own directory and loaded through mlx-taef's real from_pretrained path. No weights load
and nothing is downloaded: conftest points mlx-taef's cache at the test's directory, and the random_decoder fixture
below replaces the download step."""

import gc
import logging
import sys
import weakref
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.utils
import mlx_taef.api
import mlx_taef.download
import PIL.Image
import pytest
from _fakes import FakeZImage, LoopingFakeZImage
from mflux.models.z_image.cli import z_image_generate, z_image_turbo_generate
from mlx_taef.integrations.mflux import LivePreviewCallback

from mflux.extras.taef import _core, z_image, z_image_turbo

TURBO = z_image_turbo.ADAPTER
# Non-square on purpose: a preview decoded with height and width swapped has the wrong size.
ONE_SEED = ["--prompt", "x", "--seed", "3", "--steps", "4", "--width", "128", "--height", "64", "--output", "o.png"]


@pytest.fixture
def random_decoder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Stands in for mlx-taef's download step: writes a randomly initialised Z-Image decoder to a file and returns its
    path, so the callback's real from_pretrained -> from_pretrained_local path loads it."""
    path = tmp_path / "decoder.safetensors"

    def get_or_convert(kernel: Any, *, role: str = "decoder") -> Path:
        assert role == "decoder", f"the live preview asked mlx-taef for the {role}; it decodes only"
        model = mlx_taef.api.ZImage()
        mx.save_safetensors(str(path), dict(mlx.utils.tree_flatten(model.decoder.parameters())))
        return path

    monkeypatch.setattr(mlx_taef.download, "get_or_convert", get_or_convert)
    return path


@pytest.fixture
def real_preview(run_command, random_decoder: Path, monkeypatch: pytest.MonkeyPatch) -> list[LivePreviewCallback]:
    """Puts the real LivePreviewCallback back where run_command put the fake, and the looping model fake behind both
    mflux commands. A plain factory, not a subclass: mflux reads the signature of the call_in_loop it calls to decide
    whether to pass `denoised`. Returns the callbacks the run built."""
    built: list[LivePreviewCallback] = []

    def build(**kwargs: Any) -> LivePreviewCallback:
        preview = LivePreviewCallback(**kwargs)
        built.append(preview)
        return preview

    monkeypatch.setattr(_core, "LivePreviewCallback", build)
    monkeypatch.setattr(z_image_turbo_generate, "ZImage", LoopingFakeZImage)
    monkeypatch.setattr(z_image_generate, "ZImage", LoopingFakeZImage)
    return built


@pytest.fixture
def restore_cache_limit() -> Iterator[None]:
    """--low-ram makes mflux's MemorySaver set MLX's process-wide cache limit; put the session's limit back after."""
    previous = mx.set_cache_limit(0)
    mx.set_cache_limit(previous)
    yield
    mx.set_cache_limit(previous)


@pytest.mark.parametrize(
    ("adapter", "model_argv"),
    [
        pytest.param(TURBO, [], id="turbo"),
        pytest.param(z_image.ADAPTER, ["--model", "z-image-turbo"], id="base"),
    ],
)
def test_one_live_file_per_seed_at_the_requested_size(
    run_command, real_preview, tmp_path: Path, adapter, model_argv
) -> None:
    """Bug: the preview is never written (save_to not a Path, the callback not registered, the wrong variant), lands
    somewhere other than next to the output, or the latent reaches the decoder with height and width swapped, so a
    128x64 image gets a 64x128 preview."""
    run_command([*ONE_SEED, *model_argv], adapter)
    assert sorted(p.name for p in tmp_path.glob("o*.png")) == ["o.png", "o.preview.png"]
    with PIL.Image.open(tmp_path / "o.preview.png") as preview:
        assert preview.size == (128, 64)


def test_frames_write_one_file_per_step(run_command, real_preview, tmp_path: Path) -> None:
    """Bug: --taef-frames is not handed to the callback as numbered_frames, so the four steps overwrite one file."""
    run_command([*ONE_SEED, "--taef-frames"], TURBO)
    assert sorted(p.name for p in tmp_path.glob("o.preview*")) == [
        "o.preview_step00.png",
        "o.preview_step01.png",
        "o.preview_step02.png",
        "o.preview_step03.png",
    ]


def test_every_n_writes_at_steps_zero_n_two_n(run_command, real_preview) -> None:
    """Bug: --taef-every is not handed to the callback as every, so five steps write five previews (every=1) or one
    (the library's default of 5), not the three at steps 0, 2 and 4."""
    run_command(["--prompt", "x", "--seed", "3", "--steps", "5", "--taef-every", "2", "--output", "o.png"], TURBO)
    (preview,) = real_preview
    assert preview.saved_paths == [Path("o.preview.png")] * 3


def test_previews_do_not_change_the_latents(run_command, real_preview, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bug: the plugin changes what mflux generates (another seed, size, step count or scheduler handed to generate),
    or a preview step writes into the latent it is handed, so the run with previews ends on other latents than the
    plain mflux-generate-z-image-turbo run with the same arguments."""
    run_command(ONE_SEED, TURBO)
    monkeypatch.setattr(sys, "argv", ["mflux-generate-z-image-turbo", *ONE_SEED])
    z_image_turbo_generate.main()
    with_previews, plain = FakeZImage.instances
    (preview,) = real_preview
    assert len(preview.saved_paths) == 4  # the previews did run in the first run
    assert len(with_previews.final_latents) == len(plain.final_latents) == 1
    assert mx.array_equal(with_previews.final_latents[0], plain.final_latents[0]).item()


def test_low_ram_frees_the_transformer_with_previews_on(
    run_command, real_preview, restore_cache_limit, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug: the plugin keeps a reference to the transformer (on the preview or anywhere it holds), which defeats the
    drop mflux's MemorySaver does after the loop for one seed with --low-ram, so the weights stay in memory through
    the decode."""
    transformers: list[weakref.ref] = []

    class WatchedZImage(LoopingFakeZImage):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            transformers.append(weakref.ref(self.transformer))

    monkeypatch.setattr(z_image_turbo_generate, "ZImage", WatchedZImage)
    run_command([*ONE_SEED, "--low-ram"], TURBO)
    gc.collect()
    (model,) = FakeZImage.instances
    (transformer,) = transformers
    (preview,) = real_preview
    assert len(preview.saved_paths) == 4  # the previews were on
    assert model.transformer is None
    assert transformer() is None


def test_a_failing_decode_warns_once_and_the_image_is_saved(
    run_command, real_preview, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """Bug: the callback is built with on_error="raise" (or the library default changes under the plugin), so one
    failed preview decode ends the run with exit 1 and no image, instead of one warning and the final image."""

    def decode_image(self: Any, latents: mx.array) -> mx.array:
        raise RuntimeError("boom")

    monkeypatch.setattr(mlx_taef.api.Taef, "decode_image", decode_image)
    caplog.set_level(logging.WARNING, logger="mlx_taef.integrations.mflux")
    run_command(ONE_SEED, TURBO)
    records = [r for r in caplog.records if r.name == "mlx_taef.integrations.mflux"]
    assert [r.levelno for r in records] == [logging.WARNING]
    assert "live preview failed at step 0: boom" in records[0].getMessage()
    assert (tmp_path / "o.png").is_file()
    assert list(tmp_path.glob("o.preview*")) == []


def test_two_seeds_each_keep_their_frames(run_command, real_preview, tmp_path: Path) -> None:
    """Bug: an explicit frames path gets no per-seed suffix, so seed 2 overwrites seed 1's frames; or save_to is set
    to a str, whose numbered-frame names the library cannot build, so no frame is written at all."""
    run_command(
        ["--prompt", "x", "--seed", "1", "2", "--steps", "2", "--taef-frames", "--taef-preview", "live.png"], TURBO
    )
    assert sorted(p.name for p in tmp_path.glob("live*")) == [
        "live_seed_1_step00.png",
        "live_seed_1_step01.png",
        "live_seed_2_step00.png",
        "live_seed_2_step01.png",
    ]


def test_two_seeds_share_one_explicit_live_file(run_command, real_preview, tmp_path: Path) -> None:
    """Bug: a seed suffix is added to an explicit path for several seeds without frames, so the file a viewer is open
    on stops at seed 1. The callback's saved_paths is reset per generation, so two entries are seed 2's two steps
    written to live.png."""
    run_command(["--prompt", "x", "--seed", "1", "2", "--steps", "2", "--taef-preview", "live.png"], TURBO)
    assert sorted(p.name for p in tmp_path.glob("live*")) == ["live.png"]
    (preview,) = real_preview
    assert preview.saved_paths == [Path("live.png")] * 2
