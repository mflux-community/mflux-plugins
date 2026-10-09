"""The real mlx_teacache.apply_teacache on a fake model, without weights: the parts of the library's contract the
command relies on and FakeTeaCache cannot show."""

import gc
import json
import weakref
from pathlib import Path

import mlx_teacache
from _fakes import FakeZImage, LoopingFakeZImage
from mflux.callbacks.instances.battery_saver import BatterySaver
from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.z_image.cli import z_image_generate

from mflux.extras.teacache import _core


class _Transformer:
    """Stands in for the transformer the predict closure is built over; only its lifetime matters here."""


def test_the_predict_closure_is_the_only_holder_of_the_transformer() -> None:
    """Bug: mlx-teacache's predict factory or its handle keeps a reference to the transformer it was called with, so
    after mflux drops its predict closure (`del predict` before the after-loop callbacks), --low-ram's MemorySaver
    sets model.transformer = None and the weights still stay resident through the decode."""
    model = FakeZImage(model_config=AVAILABLE_MODELS["z-image"])
    handle = mlx_teacache.apply_teacache(model)
    factory = model._predict
    assert factory.__qualname__.startswith("make_teacache_predict_factory"), factory.__qualname__
    transformer = _Transformer()
    alive = weakref.ref(transformer)
    predict = factory(transformer)
    del transformer
    gc.collect()
    assert alive() is not None, "the closure must hold the transformer while mflux's loop runs"
    del predict
    gc.collect()
    assert alive() is None
    assert model._predict is factory and handle.stats is not None  # both still exist, and neither held it


def _run_with_the_real_library(run_command, monkeypatch, tmp_path: Path, **extra_kwargs):
    """Runs the command on LoopingFakeZImage with the real apply_teacache, passing extra_kwargs to it on top of what
    run() passes. Returns the model, the one handle and the image's JSON sidecar."""
    applied: list[tuple[LoopingFakeZImage, mlx_teacache.TeaCacheHandle]] = []

    def apply_and_keep(flux, **kwargs):
        handle = mlx_teacache.apply_teacache(flux, **kwargs, **extra_kwargs)
        applied.append((flux, handle))
        return handle

    monkeypatch.setattr(_core, "apply_teacache", apply_and_keep)
    monkeypatch.setattr(z_image_generate, "ZImage", LoopingFakeZImage)
    # mflux's battery check runs pmset; a laptop below 10 % would stop the run.
    monkeypatch.setattr(BatterySaver, "_get_battery_percentage", lambda self: None)
    out = tmp_path / "out.png"
    run_command(
        ["--prompt", "x", "--seed", "7", "--steps", "6", "--width", "64", "--height", "64", "--make-conf",
         "--output", str(out)]
    )  # fmt: skip
    ((model, handle),) = applied
    return model, handle, json.loads(out.with_suffix(".metadata.json").read_text())


def test_the_real_apply_teacache_commits_what_the_command_records(run_command, monkeypatch, tmp_path: Path) -> None:
    """Bug: the command and the real library disagree on the apply/generate/commit path that FakeTeaCache only
    imitates: run() applies TeaCache where mflux's callbacks never reach its lifecycle callback, or reads the stats
    before the wrapped generate_image commits them, so the image records nothing or another generation's counts."""
    _, handle, recorded = _run_with_the_real_library(run_command, monkeypatch, tmp_path)
    assert handle.stats.generations == 1
    decisions = [step.decision for step in handle.stats.last_generation.decisions]
    assert len(decisions) == 6, decisions
    assert decisions[0] == decisions[-1] == "forced"
    assert recorded["teacache_active_steps"] == 6
    assert recorded["teacache_skipped_steps"] == decisions.count("skipped")
    assert recorded["teacache_threshold"] == 0.12
    assert recorded["teacache_variant"] == "z-image-base"


def test_the_skipped_steps_the_real_library_decides_are_the_ones_the_image_records(
    run_command, monkeypatch, tmp_path: Path
) -> None:
    """Bug: the record counts a label the real gate never writes for a skipped step, so every image says it skipped
    0 steps. Caller coefficients are not clamped to a calibrated range: a polynomial that predicts almost no change
    keeps the accumulator under the threshold, so the real gate skips most steps between the forced first and last
    (forced, computed, skipped, skipped, skipped, forced on mlx-teacache 0.13.2)."""
    _, handle, recorded = _run_with_the_real_library(
        run_command, monkeypatch, tmp_path, coefficients=(0.0, 0.0, 0.0, 0.0, 1e-4)
    )
    decisions = [step.decision for step in handle.stats.last_generation.decisions]
    assert decisions.count("skipped") > 0, decisions
    assert recorded["teacache_skipped_steps"] == decisions.count("skipped")
    assert recorded["teacache_active_steps"] == 6


def test_the_real_lifecycle_callback_runs_before_mflux_memory_saver(run_command, monkeypatch, tmp_path: Path) -> None:
    """Bug: run() registers mflux's callbacks before apply_teacache, so MemorySaver's after-loop step (mx.clear_cache,
    or dropping the transformer with --low-ram) runs before mlx-teacache's own after-loop callback releases its cached
    residuals; those buffers go back to MLX's cache pool after the clear and stay resident through the decode.
    test_run.py checks the order with FakeTeaCache's marker; this checks where the real library registers."""
    model, _, _ = _run_with_the_real_library(run_command, monkeypatch, tmp_path)
    after_loop = model.callbacks.after_loop
    teacache_positions = [i for i, cb in enumerate(after_loop) if type(cb).__module__.startswith("mlx_teacache.")]
    saver_positions = [i for i, cb in enumerate(after_loop) if isinstance(cb, MemorySaver)]
    assert len(teacache_positions) == 1, [type(cb) for cb in after_loop]
    assert len(saver_positions) == 1, [type(cb) for cb in after_loop]
    assert teacache_positions[0] < saver_positions[0]
