"""What a finished, interrupted or failed run leaves behind."""

import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import mlx_teacache
import PIL.Image
import pytest
from _fakes import FakeZImage, commit_generation, new_handle
from _marks import SIDECAR_LEFT_OPEN_MESSAGE
from mflux.utils.exceptions import StopImageGenerationException
from mlx_teacache import TeaCacheUntestedMfluxWarning

from mflux.extras.teacache import _core

BASE = ["--prompt", "x", "--steps", "5"]


def sidecar(image_path: Path) -> dict:
    return json.loads(image_path.with_suffix(".metadata.json").read_text())


def teacache_keys(metadata: dict) -> dict:
    return {key: value for key, value in metadata.items() if key.startswith("teacache_")}


def test_a_committed_generation_records_teacache_in_the_sidecar(run_command, teacache, tmp_path) -> None:
    """Bug: the teacache_* keys are missing, count computed steps as skipped, or hold a value json.dump
    can't write (mflux's save_image swallows that error and writes no sidecar at all)."""
    out = tmp_path / "out.png"
    run_command([*BASE, "--seed", "7", "--make-conf", "--output", str(out)])
    metadata = sidecar(out)
    assert teacache_keys(metadata) == {
        "teacache_threshold": 0.12,
        "teacache_skipped_steps": 3,
        "teacache_active_steps": 5,
        "teacache_variant": "z-image-base",
        "teacache_version": mlx_teacache.__version__,
    }
    assert metadata["seed"] == 7  # mflux's own keys are untouched


def test_the_teacache_record_keeps_the_options_mflux_recorded(run_command, teacache, tmp_path) -> None:
    """Bug: the plugin replaces the image's generation_parameters with its own keys instead of merging, so the
    float32 (and compute_precision) that mflux 0.22's ZImage recorded there vanish from the sidecar and a -C
    replay of it runs in bfloat16."""
    out = tmp_path / "out.png"
    run_command([*BASE, "--seed", "7", "--float32", "--make-conf", "--output", str(out)])
    metadata = sidecar(out)
    assert metadata["float32"] is True
    assert set(teacache_keys(metadata)) == {
        "teacache_threshold",
        "teacache_skipped_steps",
        "teacache_active_steps",
        "teacache_variant",
        "teacache_version",
    }


def test_the_record_does_not_change_the_dict_mflux_handed_over(run_command, teacache, tmp_path) -> None:
    """Bug: _record_teacache updates image.generation_parameters in place, so the teacache_* keys also land in any
    image that shares that dict (GeneratedImage.get_right_half passes it on) and in mflux's own copy."""
    run_command([*BASE, "--seed", "7", "--float32", "--make-conf", "--output", str(tmp_path / "out.png")])
    (handed_over,) = FakeZImage.instances[0].returned_parameters
    assert handed_over == {"float32": True}
    assert teacache_keys(sidecar(tmp_path / "out.png")) != {}


def test_a_run_without_metadata_flags_writes_no_sidecar(run_command, teacache, tmp_path) -> None:
    """Bug: run() saves with export_json_metadata=True (or ignores --metadata / --make-conf), so every TeaCache
    image leaves a .metadata.json next to it, unlike mflux-generate-z-image."""
    run_command([*BASE, "--seed", "7", "--output", str(tmp_path / "out.png")])
    assert (tmp_path / "out.png").is_file()
    assert list(tmp_path.glob("*.metadata.json")) == []


@pytest.mark.parametrize(("argv", "shown"), [([], "0.12"), (["--teacache-threshold", "0.3"], "0.3")])
def test_the_report_line_names_skips_steps_and_the_threshold_in_force(
    run_command, teacache, capsys, argv, shown
) -> None:
    """Bug: the summary prints the flag value (None when omitted) instead of the threshold the handle
    resolved, or counts every step."""
    run_command([*BASE, "--seed", "7", *argv])
    assert f"TeaCache: skipped 3 of 5 steps (threshold {shown})" in capsys.readouterr().out


def test_an_img2img_sidecar_records_the_active_steps(run_command, teacache, tmp_path) -> None:
    """Bug: teacache_active_steps records --steps (20) instead of the steps mflux ran from its init_time_step
    (20 - int(20 * 0.6) = 8), so the sidecar claims skips out of steps that never ran."""
    ref = tmp_path / "ref.png"
    PIL.Image.new("RGB", (64, 64)).save(ref)
    out = tmp_path / "out.png"
    run_command(
        [
            "--prompt",
            "x",
            "--steps",
            "20",
            "--seed",
            "7",
            "--image",
            str(ref),
            "0.6",
            "--make-conf",
            "--output",
            str(out),
        ]
    )
    recorded = teacache_keys(sidecar(out))
    assert recorded["teacache_active_steps"] == 8
    assert recorded["teacache_skipped_steps"] == 3


def test_forced_and_missed_steps_are_not_counted_as_skips(run_command, teacache, capsys) -> None:
    """Bug: teacache_skipped_steps counts every step that was not "computed", so the two forced end steps and a
    numerical miss (all of which ran the full model) are reported as skipped: 5 of 5 here instead of 2 of 5."""
    teacache.skipped_steps = frozenset({1, 2})
    teacache.numerical_miss_step = 3
    run_command([*BASE, "--seed", "7"])
    assert "TeaCache: skipped 2 of 5 steps" in capsys.readouterr().out


def test_an_image_without_generation_parameters_still_gets_the_record() -> None:
    """Bug: _record_teacache spreads image.generation_parameters unguarded, so a model family whose image carries
    None there (mflux's GeneratedImage turns None into {}, a later family may not) fails with a TypeError after the
    image was generated, and the run exits 1 without saving it."""
    handle = new_handle(rel_l1_thresh=0.12, variant_id="z-image-base")
    commit_generation(handle, active_steps=5, skipped_steps=frozenset({1, 2, 3}))
    image = SimpleNamespace(generation_parameters=None)
    _core._record_teacache(image, handle, committed_before=0)
    assert teacache_keys(image.generation_parameters)["teacache_skipped_steps"] == 3


def test_a_sidecar_from_this_command_replays_without_its_threshold(run_command, teacache, tmp_path) -> None:
    """Bug: replaying the plugin's own sidecar with -C reads back teacache_threshold (the README says the keys are not
    read back, so a replay must use the default unless the flag is passed again) or treats the recorded model as a
    custom checkpoint and warns about the calibrated one."""
    first = tmp_path / "first.png"
    run_command([*BASE, "--seed", "7", "--teacache-threshold", "0.3", "--make-conf", "--output", str(first)])
    assert teacache_keys(sidecar(first))["teacache_threshold"] == 0.3
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # simplefilter replaces the mark's filter inside this block; put back the same narrow one.
        warnings.filterwarnings("ignore", message=SIDECAR_LEFT_OPEN_MESSAGE, category=ResourceWarning)
        run_command(["-C", str(first.with_suffix(".metadata.json")), "--output", str(tmp_path / "replay.png")])
    assert teacache.calls == [{"rel_l1_thresh": 0.3}, {"rel_l1_thresh": None}]
    replayed = FakeZImage.instances[1].generate_calls[0]
    assert (replayed["seed"], replayed["prompt"], replayed["num_inference_steps"]) == (7, "x", 5)
    assert (tmp_path / "replay.png").is_file()


def test_an_uncommitted_generation_gets_no_teacache_keys(run_command, teacache, tmp_path, capsys) -> None:
    """Bug: run() checks stats.generations > 0 or reads last_generation unconditionally, so an image whose
    generation mlx-teacache did not commit inherits the previous seed's counts."""
    teacache.uncommitted_seeds = frozenset({8})
    run_command([*BASE, "--seed", "7", "8", "--make-conf", "--output", str(tmp_path / "out.png")])
    assert teacache_keys(sidecar(tmp_path / "out_seed_7.png"))["teacache_skipped_steps"] == 3
    assert teacache_keys(sidecar(tmp_path / "out_seed_8.png")) == {}
    assert capsys.readouterr().out.count("TeaCache: skipped") == 1


def test_a_stopped_generation_is_printed_and_exits_zero(run_command, teacache, tmp_path, capsys) -> None:
    """Bug: StopImageGenerationException (Ctrl-C inside the loop, battery limit) becomes a traceback or
    exit 1, or the loop keeps going with the next seed, unlike mflux-generate-z-image."""
    FakeZImage.fail_on_seed[7] = StopImageGenerationException("Stopping image generation at step 2/5")
    run_command([*BASE, "--seed", "7", "8", "--output", str(tmp_path / "out.png")])
    out = capsys.readouterr().out
    assert "Stopping image generation at step 2/5" in out
    assert "Peak MLX memory" in out
    assert len(FakeZImage.instances[0].generate_calls) == 1
    assert list(tmp_path.glob("out*")) == []


def test_a_missing_prompt_file_is_printed_and_exits_zero(run_command, teacache, tmp_path, capsys) -> None:
    """Bug: PromptFileReadError escapes as a traceback instead of mflux's one-line message."""
    run_command(["--prompt-file", str(tmp_path / "missing.txt"), "--steps", "5", "--seed", "1"])
    assert "Prompt file does not exist" in capsys.readouterr().out


def test_a_failure_during_generation_exits_one_with_a_hint(run_command, teacache, capsys) -> None:
    """Bug: an error after apply_teacache is swallowed (exit 0, no image, no explanation) or loses its
    traceback, so nobody can file a useful report."""
    FakeZImage.fail_on_seed[7] = RuntimeError("boom")
    with pytest.raises(SystemExit) as exc:
        run_command([*BASE, "--seed", "7"])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "RuntimeError: boom" in captured.err
    assert "Run mflux-generate-z-image to generate without TeaCache" in captured.err
    assert "Peak MLX memory" in captured.out


def test_a_failure_in_apply_teacache_exits_one_before_generating(run_command, teacache) -> None:
    """Bug: an exception from apply_teacache escapes run()'s failure handling or the loop runs on an
    unpatched model."""
    teacache.error = RuntimeError("apply failed")
    with pytest.raises(SystemExit) as exc:
        run_command([*BASE, "--seed", "7"])
    assert exc.value.code == 1
    assert FakeZImage.instances[0].generate_calls == []


def test_ctrl_c_during_a_generation_propagates_as_keyboard_interrupt(run_command, teacache) -> None:
    """Bug: the failure handler catches BaseException, so Ctrl-C while generate() runs (outside mflux's own
    StopImageGenerationException path) becomes "the run failed", exit 1, instead of propagating as it does from
    mflux-generate-z-image."""
    FakeZImage.fail_on_seed[7] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run_command([*BASE, "--seed", "7"])


def test_a_newer_mflux_warns_and_still_runs(run_command, teacache, tmp_path) -> None:
    """Bug: the plugin turns mlx-teacache's untested-mflux warning into a refusal (or swallows it), against
    the rule that a newer mflux runs and reports rather than refusing to start."""
    teacache.warning = TeaCacheUntestedMfluxWarning("mflux 0.24.0 is newer than the newest mflux mlx-teacache checked")
    with pytest.warns(TeaCacheUntestedMfluxWarning):
        run_command([*BASE, "--seed", "7", "--output", str(tmp_path / "out.png")])
    assert (tmp_path / "out.png").is_file()
