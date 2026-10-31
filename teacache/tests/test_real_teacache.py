"""The real mlx_teacache.apply_teacache on a fake model, without weights: the parts of the library's contract the
command relies on and FakeTeaCache cannot show."""

import gc
import importlib.metadata
import json
import weakref
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx_teacache
import pytest
from _fakes import FakeZImage, LoopingFakeQwenImage, LoopingFakeZImage
from conftest import _clear_fakes
from mflux.callbacks.instances.battery_saver import BatterySaver
from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.qwen.cli import qwen_image_generate
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_block import QwenTransformerBlock
from mflux.models.z_image.cli import z_image_generate
from mlx_teacache import TeaCacheUncalibratedCheckpointWarning

from mflux.extras.teacache import _core, qwen_image


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
    (forced, computed, skipped, skipped, skipped, forced on mlx-teacache 0.14.0)."""
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


def test_the_real_library_warns_that_the_locked_mflux_is_newer_than_it_verified(
    run_command, monkeypatch, tmp_path: Path
) -> None:
    """Bug: the command swallows or filters mlx-teacache's untested-mflux warning (or the library stops giving it),
    so a user on mflux 0.24.0 never learns that mlx-teacache 0.14.0 was verified only up to mflux 0.23.0. The
    pyproject filter that shows this warning instead of raising it was reviewed for this pair only, so any other
    pair skips here (the mflux-main and mlx-teacache-main jobs run other pairs)."""
    mflux_version = importlib.metadata.version("mflux")
    teacache_version = importlib.metadata.version("mlx-teacache")
    if (mflux_version, teacache_version) != ("0.24.0", "0.14.0"):
        pytest.skip(
            f"checks the reviewed pair mflux 0.24.0 with mlx-teacache 0.14.0; this run has mflux {mflux_version} "
            f"with mlx-teacache {teacache_version}"
        )
    # The library warns once per process; an earlier real-library test has usually used that up.
    monkeypatch.setattr(mlx_teacache._mflux_versions, "_warned", False)
    with pytest.warns(mlx_teacache.errors.TeaCacheUntestedMfluxWarning, match=r"mflux 0\.24\.0 is newer"):
        _run_with_the_real_library(run_command, monkeypatch, tmp_path)
    assert (tmp_path / "out.png").is_file()


# Qwen-Image: the real library proxies model.transformer and runs both CFG branches through it.

QWEN_ARGV = ["--prompt", "x", "--seed", "7", "--steps", "6", "--width", "64", "--height", "64"]


@pytest.fixture
def restored_cache_limit() -> Iterator[None]:
    """MemorySaver sets MLX's cache limit for the whole process (1 GB with --low-ram); put it back for later tests."""
    before = mx.set_cache_limit(0)
    mx.set_cache_limit(before)
    yield
    mx.set_cache_limit(before)


def _run_qwen_with_the_real_library(
    run_command,
    monkeypatch,
    tmp_path: Path,
    argv: list[str],
    *,
    on_apply: Any = None,
    **extra_kwargs: Any,
) -> tuple[LoopingFakeQwenImage, mlx_teacache.TeaCacheHandle, Path]:
    """Runs the Qwen command on LoopingFakeQwenImage with the real apply_teacache (extra_kwargs on top of what run()
    passes), then on_apply(model) right after it, before run() registers mflux's callbacks. Returns the model, the one
    handle and the output path."""
    applied: list[tuple[LoopingFakeQwenImage, mlx_teacache.TeaCacheHandle]] = []

    def apply_and_keep(flux, **kwargs):
        handle = mlx_teacache.apply_teacache(flux, **kwargs, **extra_kwargs)
        applied.append((flux, handle))
        if on_apply is not None:
            on_apply(flux)
        return handle

    monkeypatch.setattr(_core, "apply_teacache", apply_and_keep)
    monkeypatch.setattr(qwen_image_generate, "QwenImage", LoopingFakeQwenImage)
    monkeypatch.setattr(BatterySaver, "_get_battery_percentage", lambda self: None)
    out = tmp_path / "out.png"
    run_command([*argv, "--output", str(out)], qwen_image.ADAPTER)
    ((model, handle),) = applied
    return model, handle, out


def _decisions(handle: mlx_teacache.TeaCacheHandle) -> list[str]:
    return [step.decision for step in handle.stats.last_generation.decisions]


def test_qwen_real_apply_commits_what_the_command_records(run_command, monkeypatch, tmp_path: Path) -> None:
    """Bug: the Qwen command and the real library disagree on the apply/generate/commit path: run() reads the stats
    before the wrapped generate_image commits them, the library records one decision per transformer call (two per
    step), or the image records another variant or threshold than the one the library ran.

    Expected decisions, from mlx-teacache 0.14.0's gate and qwen-image-2512's calibration: step 0 is forced (and only
    anchors the signal; a forced step caches nothing), step 1 has no cached residual yet so it computes and seeds,
    step 5 is forced. Steps 2 to 4 price the measured change after clamping it into the calibrated range (0.081,
    0.471): even at the range floor a step costs 0.038, and a single step reaches the 0.25 default only from a delta
    near 0.256. On this tiny transformer (fixed seeds, random weights) the modulated block-0 input moves by 0.108,
    0.115 and 0.126 on steps 2 to 4; the fit prices them 0.060, 0.066 and 0.076, which accumulate to 0.203, under
    0.25, so all three skip: forced, computed, skipped x3, forced."""
    _, handle, out = _run_qwen_with_the_real_library(run_command, monkeypatch, tmp_path, [*QWEN_ARGV, "--make-conf"])
    recorded = json.loads(out.with_suffix(".metadata.json").read_text())
    assert handle.stats.generations == 1
    rel_l1 = [step.rel_l1 for step in handle.stats.last_generation.decisions]
    assert _decisions(handle) == ["forced", "computed", "skipped", "skipped", "skipped", "forced"], rel_l1
    assert recorded["teacache_active_steps"] == 6
    assert recorded["teacache_skipped_steps"] == 3
    assert recorded["teacache_threshold"] == 0.25
    assert recorded["teacache_variant"] == "qwen-image-2512"


def test_both_branches_share_one_decision_and_skip_together(run_command, monkeypatch, tmp_path: Path) -> None:
    """Bug: the negative CFG branch takes its own gate decision, or runs the transformer blocks on a step the positive
    branch skipped, so Qwen's skip saves half of what the record says (or none) and the decisions count two per step.

    With caller coefficients (0, 0, 0, 0, 1e-4) every gated step is priced 1e-4, under the 0.20 threshold mlx-teacache
    uses for caller coefficients, and nothing clamps it: forced, computed (the seed), skipped x3, forced. Three steps
    compute, each running both blocks on both branches: 2 x 2 x 3 = 12 block calls."""
    block_calls = [0]
    real_call = QwenTransformerBlock.__call__

    def counted(self, *args: Any, **kwargs: Any) -> Any:
        block_calls[0] += 1
        return real_call(self, *args, **kwargs)

    monkeypatch.setattr(QwenTransformerBlock, "__call__", counted)
    _, handle, _ = _run_qwen_with_the_real_library(
        run_command, monkeypatch, tmp_path, QWEN_ARGV, coefficients=(0.0, 0.0, 0.0, 0.0, 1e-4)
    )
    assert _decisions(handle) == ["forced", "computed", "skipped", "skipped", "skipped", "forced"]
    assert block_calls[0] == 12


class _AliveAtBeforeLoop:
    """Registered on the model right after apply_teacache: at before_loop it records what is alive and takes a weak
    reference to one weight array of the inner transformer, for the decode-time check."""

    def __init__(self, model: LoopingFakeQwenImage, seen: dict[str, Any], weight_refs: list[Any]) -> None:
        self.model = model
        self.seen = seen
        self.weight_refs = weight_refs

    def call_before_loop(self, **kwargs: Any) -> None:
        inner = LoopingFakeQwenImage.transformer_ref and LoopingFakeQwenImage.transformer_ref()
        if inner is not None:
            self.weight_refs.append(weakref.ref(inner.img_in.weight))
        del inner
        self.seen["before_loop"] = _alive(self.model)


def _alive(model: LoopingFakeQwenImage) -> dict[str, bool]:
    gc.collect()
    ref = LoopingFakeQwenImage.transformer_ref
    return {
        "model.transformer": model.transformer is not None,
        "inner transformer": ref is not None and ref() is not None,
    }


def _run_and_watch_the_transformer(run_command, monkeypatch, tmp_path: Path, flags: list[str]) -> dict[str, Any]:
    """Runs one seed with the real library and returns what was alive at before_loop and at decode (decode_hook),
    plus whether the weight array taken at before_loop was still alive at decode. Records only: an assertion
    raised inside the run would end in run()'s own error handling."""
    seen: dict[str, Any] = {}
    weight_refs: list[Any] = []

    def register_watcher(model: LoopingFakeQwenImage) -> None:
        model.callbacks.register(_AliveAtBeforeLoop(model, seen, weight_refs))

    def at_decode(model: LoopingFakeQwenImage) -> None:
        seen["decode"] = _alive(model)
        seen["weight at decode"] = bool(weight_refs) and weight_refs[0]() is not None

    monkeypatch.setattr(LoopingFakeQwenImage, "decode_hook", at_decode)
    _run_qwen_with_the_real_library(run_command, monkeypatch, tmp_path, [*QWEN_ARGV, *flags], on_apply=register_watcher)
    seen["weight taken"] = len(weight_refs) == 1
    return seen


@pytest.mark.usefixtures("restored_cache_limit")
@pytest.mark.parametrize("flag", ["--low-ram", "--pid-decode"])
def test_low_ram_releases_the_qwen_transformer_before_decode(run_command, monkeypatch, tmp_path: Path, flag) -> None:
    """Bug: with --low-ram or --pid-decode (one seed) mflux's MemorySaver sets model.transformer = None after the loop,
    but the TeaCache proxy, its handle or its rollback still holds the real transformer, so the weights stay resident
    through the decode."""
    seen = _run_and_watch_the_transformer(run_command, monkeypatch, tmp_path, [flag])
    assert seen["weight taken"]
    assert seen["before_loop"] == {"model.transformer": True, "inner transformer": True}
    assert seen["decode"] == {"model.transformer": False, "inner transformer": False}
    assert seen["weight at decode"] is False


@pytest.mark.usefixtures("restored_cache_limit")
def test_without_either_flag_the_qwen_transformer_lives_through_decode(run_command, monkeypatch, tmp_path) -> None:
    """Bug: the release check is vacuous: the watcher's weak references die whatever mflux does (a reference taken to
    a temporary, a gc that frees a live model), so the release test passes without MemorySaver dropping anything."""
    seen = _run_and_watch_the_transformer(run_command, monkeypatch, tmp_path, [])
    assert seen["weight taken"]
    assert seen["decode"] == {"model.transformer": True, "inner transformer": True}
    assert seen["weight at decode"] is True


def test_real_library_warns_once_for_an_unknown_qwen_checkpoint(run_command, monkeypatch, tmp_path: Path) -> None:
    """Bug: for a local folder the plugin warns and the real library warns again (or neither does), so the user gets
    two warnings for one unverified checkpoint, or none, or one that does not name the checkpoint the settings were
    tuned on."""
    with pytest.warns(TeaCacheUncalibratedCheckpointWarning) as caught:
        _run_qwen_with_the_real_library(run_command, monkeypatch, tmp_path, [*QWEN_ARGV, "--model", "./weights"])
    messages = [str(w.message) for w in caught if issubclass(w.category, TeaCacheUncalibratedCheckpointWarning)]
    assert len(messages) == 1, messages
    assert "Qwen/Qwen-Image-2512" in messages[0]


def test_the_looping_qwen_fakes_class_state_is_cleared_between_tests(monkeypatch) -> None:
    """Bug: the per-test reset leaves LoopingFakeQwenImage's class-level decode_hook or transformer_ref set, so a test
    that set the hook without monkeypatch makes every later looping run call it."""
    monkeypatch.setattr(LoopingFakeQwenImage, "decode_hook", lambda model: None)
    monkeypatch.setattr(LoopingFakeQwenImage, "transformer_ref", lambda: None)
    _clear_fakes()
    assert LoopingFakeQwenImage.decode_hook is None
    assert LoopingFakeQwenImage.transformer_ref is None
