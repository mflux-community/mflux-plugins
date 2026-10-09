"""Everything the plugin refuses happens before any weights load, as exit 2 with a usable message."""

import dataclasses
import gc
import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import mlx.core as mx
import PIL.Image
import pytest
from _fakes import FakeZImage
from _marks import SIDECAR_LEFT_OPEN_BY_MFLUX, SIDECAR_LEFT_OPEN_MESSAGE
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.compute_precision.compute_precision import ComputePrecision
from mflux.models.z_image.cli import z_image_generate
from mlx_teacache import TeaCacheUncalibratedCheckpointWarning, VariantInfo

from mflux.extras.teacache import _core, z_image

PLAIN_HINT = "Run mflux-generate-z-image to generate without TeaCache"


def too_short(has: object) -> str:
    """The step refusal as printed, for a run with `has` active steps."""
    return (
        "TeaCache needs at least 3 denoising steps: it always computes the first and the last one. "
        f"This run has {has}. Use more --steps (or a lower --image strength), "
        "or run mflux-generate-z-image without TeaCache."
    )


def refused(run_command, argv, capsys, adapter=None) -> str:
    with pytest.raises(SystemExit) as exc:
        run_command(argv, adapter)
    assert exc.value.code == 2
    assert FakeZImage.instances == [], "weights were loaded before the refusal"
    return capsys.readouterr().err


def step_cache_adapter() -> _core.Adapter:
    def build_parser() -> CommandLineParser:
        parser = z_image.build_parser()
        parser.add_step_cache_arguments()
        return parser

    return dataclasses.replace(z_image.ADAPTER, build_parser=build_parser)


def reference_png(tmp_path: Path) -> Path:
    path = tmp_path / "ref.png"
    PIL.Image.new("RGB", (64, 64)).save(path)
    return path


@pytest.mark.parametrize("value", ["-0.1", "0", "1.01", "nan", "inf"])
def test_a_threshold_outside_zero_to_one_is_refused(run_command, teacache, capsys, value) -> None:
    """Bug: the range check uses >= 0 (0 disables TeaCache yet the run claims it), lets 1.01 through,
    misses NaN because NaN fails every comparison, or drops the way out (the plain command)."""
    err = refused(run_command, ["--prompt", "x", "--teacache-threshold", value], capsys)
    assert "--teacache-threshold must be greater than 0 and at most 1" in err
    assert PLAIN_HINT in err
    assert teacache.calls == []


@pytest.mark.parametrize(("value", "expected"), [("0.01", 0.01), ("1", 1.0)])
def test_the_edges_of_the_threshold_range_run(run_command, teacache, value, expected) -> None:
    """Bug: the range check is written 0 < t < 1 (refusing 1.0) or t > 0.01."""
    run_command(["--prompt", "x", "--seed", "1", "--steps", "5", "--teacache-threshold", value])
    assert teacache.calls == [{"rel_l1_thresh": expected}]


def test_the_installed_command_refuses_a_zero_threshold_before_loading(tmp_path: Path) -> None:
    """Bug: the console script reaches something other than z_image.main() -> run(ADAPTER) (a stale entry point, a
    main() that skips run), so the real command loses the pre-load refusals the in-process tests exercise."""
    script = Path(sys.executable).with_name("mflux-generate-z-image-teacache")
    assert script.is_file(), f"console script not installed next to {sys.executable}; run uv sync"
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HOME": str(tmp_path / "empty-hf-home")}
    result = subprocess.run(
        [str(script), "--prompt", "x", "--teacache-threshold", "0"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert result.returncode == 2, result.stderr
    assert "--teacache-threshold must be greater than 0 and at most 1" in result.stderr
    assert PLAIN_HINT in result.stderr


def test_a_model_of_another_family_is_refused(run_command, teacache, capsys) -> None:
    """Bug: ModelConfigError from mflux's validate() escapes as a traceback (exit 1) instead of a
    parser error."""
    err = refused(run_command, ["--prompt", "x", "--model", "dev"], capsys)
    assert "'dev' is not Tongyi-MAI/Z-Image" in err


def test_the_distilled_family_member_is_refused_and_supported_names_listed(run_command, teacache, capsys) -> None:
    """Bug: the plugin trusts mflux's family list (turbo validates on this command) and loads turbo,
    or the message lists turbo among the supported names."""
    err = refused(run_command, ["--prompt", "x", "--model", "z-image-turbo"], capsys)
    assert "TeaCache doesn't support z-image-turbo yet" in err
    listed = err.split("Supported on this command: ")[1].split(". ")[0].split(", ")
    assert "z-image" in listed
    assert "z-image-turbo" not in listed
    assert PLAIN_HINT in err


def test_mflux_still_leaves_a_replayed_sidecar_open(monkeypatch, tmp_path: Path) -> None:
    """Bug: mflux closes the -C file now, and SIDECAR_LEFT_OPEN_BY_MFLUX stays on these tests, where it would hide a
    sidecar this package itself leaves open. Red here means: delete the filter (tests/_marks.py) and its uses."""
    sidecar = tmp_path / "replay.metadata.json"
    sidecar.write_text(json.dumps({"model": "Tongyi-MAI/Z-Image", "prompt": "p", "seed": 3, "steps": 5}))
    argv = ["-C", str(sidecar)]
    monkeypatch.setattr(sys, "argv", ["mflux-generate-z-image", *argv])
    with pytest.warns(ResourceWarning) as caught:
        z_image_generate.build_parser().parse_args()
        gc.collect()
    assert [str(w.message) for w in caught if re.match(SIDECAR_LEFT_OPEN_MESSAGE, str(w.message))] != []


@SIDECAR_LEFT_OPEN_BY_MFLUX
def test_a_turbo_sidecar_is_refused_before_load(run_command, teacache, capsys, tmp_path: Path) -> None:
    """Bug: a -C sidecar from mflux-generate-z-image-turbo (model = the turbo repo id) is treated as a
    custom Z-Image checkpoint and loaded."""
    sidecar = tmp_path / "turbo.metadata.json"
    sidecar.write_text(json.dumps({"model": "Tongyi-MAI/Z-Image-Turbo", "prompt": "p", "seed": 3, "steps": 9}))
    err = refused(run_command, ["-C", str(sidecar)], capsys)
    assert "TeaCache doesn't support Tongyi-MAI/Z-Image-Turbo yet" in err


def test_a_model_without_a_default_threshold_is_refused(run_command, teacache, capsys, monkeypatch) -> None:
    """Bug: a distilled variant (default_thresh None, e.g. FLUX.2 Klein 4B under a future adapter)
    runs with the library's fallback threshold and skips nothing."""
    distilled = VariantInfo(
        variant_id="flux2-klein-4b",
        display_name="FLUX.2 Klein 4B",
        default_thresh=None,
        model_names=("flux2-klein-4b",),
        calibrated=True,
    )
    monkeypatch.setattr(_core, "match_variant", lambda model_config, pipeline_class: distilled)
    err = refused(run_command, ["--prompt", "x"], capsys)
    assert "FLUX.2 Klein 4B is a distilled few-step model" in err
    assert "Run mflux-generate-z-image instead" in err


@pytest.mark.parametrize(
    "argv",
    [["--step-cache-ratio", "0.25"], ["--teacache-ratio", "0.25"]],
)
def test_mflux_step_cache_is_refused_where_the_parser_has_it(run_command, teacache, capsys, argv) -> None:
    """Bug: the plugin lets mflux's step cache and TeaCache both skip steps in one run, or refuses without
    saying why."""
    err = refused(run_command, ["--prompt", "x", *argv], capsys, step_cache_adapter())
    assert "--step-cache-ratio" in err
    assert "can't run together" in err
    assert "Remove --step-cache-ratio (or --teacache-ratio) from the command line" in err


@SIDECAR_LEFT_OPEN_BY_MFLUX
def test_a_step_cache_ratio_from_a_sidecar_is_refused(run_command, teacache, capsys, tmp_path: Path) -> None:
    """Bug: the check reads only the command line, so a -C sidecar that recorded a step-cache ratio
    turns mflux's step cache on next to TeaCache; or the refusal tells the user to drop a flag they never typed."""
    sidecar = tmp_path / "sc.metadata.json"
    sidecar.write_text(
        json.dumps({"model": "Tongyi-MAI/Z-Image", "prompt": "p", "seed": 3, "steps": 20, "step_cache_ratio": 0.25})
    )
    err = refused(run_command, ["-C", str(sidecar)], capsys, step_cache_adapter())
    assert "--step-cache-ratio" in err
    assert "can't run together" in err
    # "Drop the flag" can't be followed when the value came from the file: the hint names the file's key.
    assert "the step_cache_ratio key from the -C file" in err


def test_a_parser_with_step_cache_runs_when_it_is_unset(run_command, teacache) -> None:
    """Bug: the check refuses whenever the parser has the option (hasattr) instead of when it is set."""
    run_command(["--prompt", "x", "--seed", "1", "--steps", "5"], step_cache_adapter())
    assert len(FakeZImage.instances) == 1


def test_the_z_image_command_does_not_take_a_step_cache_flag(run_command, teacache, capsys) -> None:
    """Bug: the plugin adds mflux's step-cache options to a command whose mflux parser lacks them."""
    err = refused(run_command, ["--prompt", "x", "--step-cache-ratio", "0.25"], capsys)
    assert "unrecognized arguments" in err


@pytest.mark.parametrize(("steps", "runs"), [("2", False), ("3", True)])
def test_too_few_steps_are_refused(run_command, teacache, capsys, steps, runs) -> None:
    """Bug: the window check is off by one: it refuses 3 steps (accepted, though they skip nothing) or lets 2 through,
    where the first and last forced steps leave no step to gate."""
    argv = ["--prompt", "x", "--seed", "1", "--steps", steps]
    if runs:
        run_command(argv)
        assert len(FakeZImage.instances) == 1
    else:
        assert too_short(2) in refused(run_command, argv, capsys)


def test_the_step_refusal_is_the_plugins_own_sentence(run_command, teacache, capsys) -> None:
    """Bug: the refusal prints mlx-teacache's internal text (skip_first_n_steps + skip_last_n_steps, active_num_steps=)
    instead of saying in plain words why the run is too short and what to do."""
    err = refused(run_command, ["--prompt", "x", "--seed", "1", "--steps", "2"], capsys)
    assert (
        "TeaCache needs at least 3 denoising steps: it always computes the first and the last one. This run has 2. "
        "Use more --steps (or a lower --image strength), or run mflux-generate-z-image without TeaCache."
    ) in err
    assert "skip_first" not in err
    assert "active_num_steps" not in err


@pytest.mark.parametrize(
    ("steps", "strength", "this_run"),
    [
        ("2", None, "This run has 2."),
        ("4", "0.5", "This run has 2 of its 4 --steps."),
        ("3", "1.0", "This run has 0 of its 3 --steps."),
        ("-1", None, "This run has no valid step count (--steps -1)."),
    ],
    ids=["steps-2", "steps-4-strength-0.5", "steps-3-strength-1.0", "steps-minus-one"],
)
def test_the_step_refusal_says_how_many_steps_the_run_has(
    run_command, teacache, capsys, tmp_path, steps, strength, this_run
) -> None:
    """Bug: the count sentence hides that an --image strength cut the run below --steps ("This run has 2." for
    --steps 4), says "none" for 0 active steps, or prints a negative step count as if it were a count."""
    argv = ["--prompt", "x", "--seed", "1", "--steps", steps]
    if strength is not None:
        argv += ["--image", str(reference_png(tmp_path)), strength]
    err = refused(run_command, argv, capsys)
    assert (
        "TeaCache needs at least 3 denoising steps: it always computes the first and the last one. "
        f"{this_run} Use more --steps (or a lower --image strength), or run mflux-generate-z-image without TeaCache."
    ) in err


@pytest.mark.parametrize(("strength", "runs"), [("0.9", False), ("0.85", True)])
def test_a_short_img2img_window_is_refused(run_command, teacache, capsys, tmp_path, strength, runs) -> None:
    """Bug: the window is computed from --steps alone, ignoring that img2img starts at
    max(1, int(steps * strength)): 20 steps at 0.9 leave 2 active steps, at 0.85 leave 3."""
    argv = ["--prompt", "x", "--seed", "1", "--steps", "20", "--image", str(reference_png(tmp_path)), strength]
    if runs:
        run_command(argv)
        assert FakeZImage.instances[0].generate_calls[0]["image_strength"] == 0.85
    else:
        err = refused(run_command, argv, capsys)
        assert too_short("2 of its 20 --steps") in err


def test_zero_active_steps_are_refused(run_command, teacache, capsys, tmp_path) -> None:
    """Bug: strength 1.0 (mflux starts at the last step, so no denoising runs) slips through as a no-op run under a
    TeaCache command, unlike every other too-short window."""
    argv = ["--prompt", "x", "--seed", "1", "--steps", "20", "--image", str(reference_png(tmp_path)), "1.0"]
    err = refused(run_command, argv, capsys)
    assert too_short("0 of its 20 --steps") in err


@pytest.mark.parametrize("img2img", [False, True], ids=["steps-minus-one", "steps-zero-img2img"])
def test_a_negative_active_step_count_is_refused(run_command, teacache, capsys, tmp_path, img2img) -> None:
    """Bug: only InvalidStepWindowError is turned into a refusal, so the TeaCacheValueError mlx-teacache raises for a
    negative count (--steps -1; --steps 0 with --image, where mflux starts at step max(1, 0) = 1) escapes as a
    traceback with exit 1."""
    steps = "0" if img2img else "-1"
    argv = ["--prompt", "x", "--seed", "1", "--steps", steps]
    if img2img:
        argv += ["--image", str(reference_png(tmp_path)), "0.5"]
    err = refused(run_command, argv, capsys)
    assert too_short(f"no valid step count (--steps {steps})") in err
    assert "active_num_steps" not in err


def test_the_adapters_active_steps_hook_decides_the_window(run_command, teacache, capsys) -> None:
    """Bug: run() always uses the Config-based default, so a future command whose schedule differs (a hook that
    knows its real step count) is checked against the wrong number."""
    two_steps = dataclasses.replace(z_image.ADAPTER, active_steps=lambda args, model_config: 2)
    err = refused(run_command, ["--prompt", "x", "--seed", "1", "--steps", "20"], capsys, two_steps)
    assert too_short("2 of its 20 --steps") in err


def test_a_refused_run_prints_no_mflux_option_warnings(run_command, teacache, capsys) -> None:
    """Bug: mflux's option warnings (the after_checks hook) run before the TeaCache checks, so a refused
    turbo run first warns that --guidance is ignored instead of printing one clear refusal."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        refused(run_command, ["--prompt", "x", "--model", "z-image-turbo", "--guidance", "3"], capsys)


def test_a_custom_checkpoint_warns_and_runs(run_command, teacache) -> None:
    """Bug: a finetune or third-party repo runs with Z-Image's coefficients without saying the skips
    were never checked on it."""
    with pytest.warns(TeaCacheUncalibratedCheckpointWarning, match="someorg/my-zimage-finetune"):
        run_command(["--prompt", "x", "--seed", "1", "--steps", "5", "--model", "someorg/my-zimage-finetune"])
    assert FakeZImage.instances[0].init_kwargs["model_path"] == "someorg/my-zimage-finetune"


def test_the_calibrated_repo_id_is_not_a_custom_checkpoint(run_command, teacache) -> None:
    """Bug: the warning fires whenever model_path is set, so --model Tongyi-MAI/Z-Image (and every
    replayed Z-Image sidecar, whose model key is that repo id) warns about the calibrated checkpoint."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run_command(["--prompt", "x", "--seed", "1", "--steps", "5", "--model", "Tongyi-MAI/Z-Image"])
    assert len(FakeZImage.instances) == 1


ORG_BASE_COPIES = [
    "mflux-community/z-image-base-mflux-q3",
    "mflux-community/z-image-base-mflux-q4",
    "mflux-community/z-image-base-mflux-q5",
    "mflux-community/z-image-base-mflux-q6",
    "mflux-community/z-image-base-mflux-q8",
    "mflux-community/z-image-base-mflux-bf16",
]


@pytest.mark.parametrize("repo", ORG_BASE_COPIES)
def test_a_known_copy_of_the_calibrated_checkpoint_does_not_warn(run_command, teacache, repo) -> None:
    """Bug: the model-path check compares only against Tongyi-MAI/Z-Image, so the org's own copies of that model at
    other precisions (each declares base_model: Tongyi-MAI/Z-Image on the Hub) warn on every run; or one id is
    misspelled in the adapter's list."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run_command(["--prompt", "x", "--seed", "1", "--steps", "5", "--model", repo])
    assert FakeZImage.instances[0].init_kwargs["model_path"] == repo


def test_no_turbo_copy_is_listed_as_a_copy_of_the_calibrated_checkpoint() -> None:
    """Bug: a z-image-turbo-* repo (Turbo, a different model) is added to the adapter's known copies of
    Tongyi-MAI/Z-Image, so a Turbo checkpoint forced through with --base-model z-image runs without the warning."""
    assert [repo for repo in z_image.ADAPTER.calibrated_copies if "turbo" in repo.lower()] == []


TURBO_COPIES = [
    "mflux-community/z-image-turbo-mflux-q3",
    "mflux-community/z-image-turbo-mflux-q4",
    "mflux-community/z-image-turbo-mflux-q5",
    "mflux-community/z-image-turbo-mflux-q6",
    "mflux-community/z-image-turbo-mflux-q8",
    "mflux-community/z-image-turbo-mflux-bf16",
]


@pytest.mark.parametrize(
    "repo",
    [
        # Turbo is a different model: mflux resolves these names to the Turbo config and the command refuses them, so
        # --base-model z-image forces each through to the checkpoint check.
        *TURBO_COPIES,
        "someorg/z-image-base-mflux-q4",
    ],
)
def test_a_lookalike_of_a_known_copy_still_warns(run_command, teacache, repo) -> None:
    """Bug: the known-copy check matches by substring or suffix ("z-image-base-mflux", "mflux-q4") or ignores the
    owner, so a Turbo copy loaded as base, or another owner's repo with the same name, runs unannounced."""
    with pytest.warns(TeaCacheUncalibratedCheckpointWarning, match=re.escape(repo)):
        run_command(["--prompt", "x", "--seed", "1", "--steps", "5", "--model", repo, "--base-model", "z-image"])
    assert FakeZImage.instances[0].init_kwargs["model_path"] == repo


def test_the_custom_checkpoint_warning_says_what_is_and_is_not_known(run_command, teacache) -> None:
    """Bug: the warning states as fact that the weights differ, calls another precision of the model fine although
    only one 8-bit build was measured, drops the note that TeaCache still runs, or loses the repo names, so the user
    can't tell whether to act."""
    with pytest.warns(TeaCacheUncalibratedCheckpointWarning) as caught:
        run_command(["--prompt", "x", "--seed", "1", "--steps", "5", "--model", "someorg/my-zimage-finetune"])
    messages = [str(w.message) for w in caught if issubclass(w.category, TeaCacheUncalibratedCheckpointWarning)]
    assert messages == [
        "someorg/my-zimage-finetune may not be Tongyi-MAI/Z-Image, the checkpoint TeaCache's settings were tuned on. "
        "Skip counts and image quality on it are unmeasured. TeaCache still runs."
    ]


def test_an_uncalibrated_checkpoint_warns_once_and_runs(run_command, teacache, monkeypatch) -> None:
    """Bug: a supported model on a checkpoint its coefficients were not fitted on (VariantInfo.calibrated False; on a
    future Qwen command, Qwen-Image-2512) is refused, runs silently, or warns twice. Z-Image is always calibrated, so
    a stub stands in for match_variant here."""
    uncalibrated = VariantInfo(
        variant_id="z-image-base",
        display_name="Z-Image base",
        default_thresh=0.12,
        model_names=("z-image", "zimage"),
        calibrated=False,
    )
    monkeypatch.setattr(_core, "match_variant", lambda model_config, pipeline_class: uncalibrated)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_command(["--prompt", "x", "--seed", "1", "--steps", "5", "--model", "someorg/my-zimage-finetune"])
    messages = [str(w.message) for w in caught if issubclass(w.category, TeaCacheUncalibratedCheckpointWarning)]
    others = [
        f"{w.category.__name__}: {w.message}" for w in caught if w.category is not TeaCacheUncalibratedCheckpointWarning
    ]
    assert others == []
    assert messages == [
        "someorg/my-zimage-finetune is not the checkpoint Z-Image base's TeaCache settings were tuned on. "
        "Skip counts and image quality on it are unmeasured. TeaCache still runs."
    ]
    assert len(FakeZImage.instances) == 1


def test_a_family_whose_library_checks_the_checkpoint_gets_no_second_warning(run_command, teacache) -> None:
    """Bug: the plugin's own model-path warning also runs where mlx-teacache's VariantInfo.calibrated already covers
    the checkpoint (the future Qwen command), so a mirror the library accepts as calibrated still warns, and an
    uncalibrated one warns twice."""
    adapter = dataclasses.replace(z_image.ADAPTER, library_checks_checkpoint=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run_command(["--prompt", "x", "--seed", "1", "--steps", "5", "--model", "someorg/my-zimage-finetune"], adapter)
    assert FakeZImage.instances[0].init_kwargs["model_path"] == "someorg/my-zimage-finetune"


FLOAT16_UNMEASURED = "have not been measured with --compute-precision float16; the run uses it as asked"


def test_float16_compute_warns_and_runs_with_it(run_command, teacache) -> None:
    """Bug: --compute-precision float16 (mflux 0.22) runs under TeaCache without saying its skips and image quality
    were never measured with float16 compute, or the plugin drops the option on the way to mflux's load()."""
    with pytest.warns(UserWarning, match=FLOAT16_UNMEASURED):
        run_command(["--prompt", "x", "--seed", "1", "--steps", "5", "--compute-precision", "float16"])
    assert FakeZImage.instances[0].init_kwargs["compute_precision"] == mx.float16


@SIDECAR_LEFT_OPEN_BY_MFLUX
def test_float16_compute_from_a_sidecar_warns(run_command, teacache, tmp_path: Path) -> None:
    """Bug: the check reads only the command line, so a -C sidecar that recorded float16 compute replays it under
    TeaCache without the warning."""
    sidecar = tmp_path / "fp16.metadata.json"
    sidecar.write_text(
        json.dumps(
            {"model": "Tongyi-MAI/Z-Image", "prompt": "p", "seed": 3, "steps": 5, "compute_precision": "float16"}
        )
    )
    with pytest.warns(UserWarning, match=FLOAT16_UNMEASURED):
        run_command(["-C", str(sidecar)])
    assert FakeZImage.instances[0].init_kwargs["compute_precision"] == mx.float16


def test_the_default_compute_precision_does_not_warn(run_command, teacache) -> None:
    """Bug: the float16 check fires whenever the parser has the option (hasattr) instead of when float16 is chosen."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run_command(["--prompt", "x", "--seed", "1", "--steps", "5"])
    assert FakeZImage.instances[0].init_kwargs["compute_precision"] is None


def test_a_refused_float16_run_prints_only_the_refusal(run_command, teacache, capsys) -> None:
    """Bug: the float16 warning is issued before TeaCache's refusals, so a refused run warns about a run that never
    happens instead of printing one clear refusal."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        err = refused(run_command, ["--prompt", "x", "--compute-precision", "float16", "--steps", "2"], capsys)
    assert too_short(2) in err


def test_any_non_default_compute_precision_warns_naming_it(run_command, teacache, monkeypatch) -> None:
    """Bug: the check matches the literal "float16", so a compute precision a later mflux adds (simulated here by
    adding a choice to mflux's own ComputePrecision.CHOICES) runs under TeaCache unmeasured and unannounced."""
    monkeypatch.setitem(ComputePrecision.CHOICES, "bfloat16", mx.bfloat16)
    with pytest.warns(UserWarning, match="have not been measured with --compute-precision bfloat16; the run uses it"):
        run_command(["--prompt", "x", "--seed", "1", "--steps", "5", "--compute-precision", "bfloat16"])
    assert FakeZImage.instances[0].init_kwargs["compute_precision"] == mx.bfloat16
