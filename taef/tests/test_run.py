"""Both commands run mflux's own Z-Image and Z-Image Turbo steps: load, callbacks, generate and save."""

import dataclasses
import gc
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import PIL.Image
import pytest
from _fakes import FakeLivePreview, FakeZImage, LoopingFakeZImage
from _marks import SIDECAR_LEFT_OPEN_BY_MFLUX, SIDECAR_LEFT_OPEN_MESSAGE
from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.common.resolution.actions import PathAction, Rule
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.z_image.cli import z_image_generate, z_image_turbo_generate
from mflux.utils.exceptions import ModelConfigError, StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage

from mflux.extras.taef import z_image, z_image_turbo

BASE = ["--prompt", "a puffin", "--steps", "4"]
# Every command the plugin ships. The flow is _core.run for both, so the tests of that flow run each command.
ADAPTERS = [pytest.param(z_image_turbo.ADAPTER, id="turbo"), pytest.param(z_image.ADAPTER, id="base")]
# (adapter, its own command, the plain mflux command its messages name), spelled out per command.
COMMANDS = [
    pytest.param(
        z_image_turbo.ADAPTER, "mflux-generate-z-image-turbo-taef", "mflux-generate-z-image-turbo", id="turbo"
    ),
    pytest.param(z_image.ADAPTER, "mflux-generate-z-image-taef", "mflux-generate-z-image", id="base"),
]


@pytest.fixture
def ref_png(tmp_path: Path) -> Path:
    path = tmp_path / "ref.png"
    PIL.Image.new("RGB", (64, 32)).save(path)
    return path


@pytest.fixture
def lora_file(tmp_path: Path) -> Path:
    path = tmp_path / "style.safetensors"  # parse_args resolves local LoRA paths; a missing one is exit 2
    path.write_bytes(b"")
    return path


def test_every_flag_reaches_mflux_steps_unchanged(run_command, tmp_path, ref_png, lora_file) -> None:
    """Bug: run() builds its own model or generate call instead of calling mflux's load()/generate(), a flag (LoRA
    scale, img2img strength, PiD, size, --float32) is dropped or rewritten on the way, or a seed is skipped."""
    run_command(
        [
            "--prompt", "a puffin",
            "--seed", "7", "8",
            "--steps", "20",
            "--width", "320", "--height", "192",
            "--image", str(ref_png), "0.55",
            "--guidance", "2.5",
            "--negative-prompt", "blurry",
            "--scheduler", "linear",
            "--pid-decode", "--pid-degrade-sigma", "0.2",
            "--lora", str(lora_file), "0.5",
            "-q", "8",
            "--float32",
            "--make-conf",
            "--output", str(tmp_path / "out.png"),
        ],
    )  # fmt: skip
    (model,) = FakeZImage.instances
    assert model.init_kwargs == {
        "model_config": AVAILABLE_MODELS["z-image"],
        "quantize": 8,
        "model_path": None,
        "float32": True,
        "compute_precision": None,
        "lora_paths": [str(lora_file)],
        "lora_scales": [0.5],
        "bake_lora": True,
    }
    expected = {
        "prompt": "a puffin",
        "width": 320,
        "height": 192,
        "guidance": 2.5,
        "image_path": ref_png,
        "num_inference_steps": 20,
        "image_strength": 0.55,
        "scheduler": "linear",
        "negative_prompt": "blurry",
        "pid_decode": True,
        "pid_degrade_sigma": 0.2,
    }
    assert model.generate_calls == [{"seed": 7, **expected}, {"seed": 8, **expected}]
    assert (tmp_path / "out_seed_7.png").is_file()
    assert (tmp_path / "out_seed_8.metadata.json").is_file()


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_mflux_registers_its_usual_callbacks(run_command, adapter) -> None:
    """Bug: run() skips CallbackManager.register_callbacks, so mflux's BatterySaver and MemorySaver never run and the
    run neither stops at the battery limit nor frees memory as the plain mflux command does (no
    --stepwise-image-output-dir here, so no StepwiseHandler)."""
    run_command([*BASE, "--seed", "5"], adapter)
    registry = FakeZImage.instances[0].callbacks
    # The preview comes first (test_preview_runs_before_mflux_callbacks); mflux's own follow it.
    assert [type(c).__name__ for c in registry.before_loop] == ["FakeLivePreview", "BatterySaver", "MemorySaver"]
    assert [type(c).__name__ for c in registry.in_loop] == ["FakeLivePreview", "MemorySaver"]
    assert [type(c).__name__ for c in registry.after_loop] == ["MemorySaver"]
    assert registry.interrupt == []


def test_one_image_is_saved_per_seed(run_command, tmp_path: Path) -> None:
    """Bug: the loop saves only the last seed, or ignores --output's {seed} field. With more than one seed, mflux's
    parse_args itself appends _seed_{seed} to the output stem, hence out_3_seed_3.png."""
    run_command([*BASE, "--seed", "3", "4", "--output", "out_{seed}.png"])
    assert sorted(p.name for p in tmp_path.glob("out_*.png")) == ["out_3_seed_3.png", "out_4_seed_4.png"]


@pytest.mark.parametrize(
    ("adapter", "warned"),
    [
        # Each option makes that command's after_checks warn when the run is not refused.
        pytest.param(z_image_turbo.ADAPTER, ["--guidance", "3.5"], id="turbo"),
        pytest.param(z_image.ADAPTER, ["--negative-prompt", "blurry"], id="base"),
    ],
)
def test_a_refused_run_prints_no_mflux_option_warnings(run_command, capsys, adapter, warned) -> None:
    """Bug: run() calls after_checks before the preview checks, so a refused run first warns that --guidance is
    ignored (Turbo) or that --negative-prompt has no effect (base) instead of printing one clear refusal."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(SystemExit) as exit_info:
            run_command([*BASE, *warned, "--taef-every", "0"], adapter)
    assert exit_info.value.code == 2
    assert "--taef-every must be 1 or more, got 0." in capsys.readouterr().err


def test_mflux_own_option_warnings_still_fire(run_command) -> None:
    """Bug: run() never calls the after_checks hook, so a negative prompt that has no effect at the default guidance
    goes unmentioned, unlike in mflux-generate-z-image."""
    with pytest.warns(UserWarning, match="--negative-prompt has no effect"):
        run_command([*BASE, "--seed", "1", "--negative-prompt", "blurry"])


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_an_adapter_without_pre_load_warnings_runs(run_command, adapter) -> None:
    """Bug: run() calls after_checks unconditionally, so an adapter whose mflux command has no pre-load warning must
    pass a stub function, or crashes with 'NoneType' object is not callable."""
    run_command([*BASE, "--seed", "1"], dataclasses.replace(adapter, after_checks=None))
    assert len(FakeZImage.instances[0].generate_calls) == 1


def test_a_model_config_error_becomes_exit_2(run_command, capsys) -> None:
    """Bug: a ModelConfigError from validate escapes as a traceback instead of mflux's usage error, or the preview
    decoder is built before validate refuses."""

    def failing_validate(args):
        raise ModelConfigError("no such model")

    with pytest.raises(SystemExit) as exit_info:
        run_command(BASE, dataclasses.replace(z_image.ADAPTER, validate=failing_validate))
    assert exit_info.value.code == 2
    assert "no such model" in capsys.readouterr().err
    assert FakeZImage.instances == []
    assert FakeLivePreview.instances == []


def test_a_failure_after_load_exits_1_with_the_traceback(run_command, capsys, tmp_path: Path) -> None:
    """Bug: an exception during generation is swallowed (exit 0, no image) or loses its traceback, or an image is
    still saved for the failed seed."""
    FakeZImage.fail_on_seed[9] = RuntimeError("boom")
    with pytest.raises(SystemExit) as exit_info:
        run_command([*BASE, "--seed", "9"])
    assert exit_info.value.code == 1
    captured = capsys.readouterr()
    assert "RuntimeError: boom" in captured.err
    assert (
        "mflux-generate-z-image-taef: error: the run failed (traceback above). "
        "Run mflux-generate-z-image to generate without previews." in captured.err
    )
    assert "Peak MLX memory" in captured.out
    assert list(tmp_path.glob("*.png")) == []


def test_a_stopped_generation_is_printed_and_exits_zero(run_command, tmp_path, capsys) -> None:
    """Bug: StopImageGenerationException (Ctrl-C inside the loop, battery limit) becomes a traceback or exit 1, or the
    loop keeps going with the next seed, unlike mflux-generate-z-image."""
    FakeZImage.fail_on_seed[7] = StopImageGenerationException("Stopping image generation at step 2/4")
    run_command([*BASE, "--seed", "7", "8", "--output", str(tmp_path / "out.png")])
    out = capsys.readouterr().out
    assert "Stopping image generation at step 2/4" in out
    assert "Peak MLX memory" in out
    assert len(FakeZImage.instances[0].generate_calls) == 1
    assert list(tmp_path.glob("out*")) == []


def test_a_missing_prompt_file_is_printed_and_exits_zero(run_command, tmp_path, capsys) -> None:
    """Bug: PromptFileReadError escapes as a traceback instead of mflux's one-line message."""
    run_command(["--prompt-file", str(tmp_path / "missing.txt"), "--steps", "4", "--seed", "1"])
    assert "Prompt file does not exist" in capsys.readouterr().out


def test_a_run_without_metadata_writes_no_sidecar(run_command, tmp_path: Path) -> None:
    """Bug: run() saves with export_json_metadata=True (or ignores --metadata), so every image leaves a
    .metadata.json next to it, unlike mflux-generate-z-image."""
    run_command([*BASE, "--seed", "6", "--output", "img.png"])
    assert (tmp_path / "img.png").is_file()
    assert list(tmp_path.glob("*.metadata.json")) == []


def test_turbo_command_loads_turbo_and_saves(run_command, tmp_path: Path) -> None:
    """Bug: the Turbo adapter names the base command's steps (ZImageCommand.load), so the Turbo command loads the
    z-image config, or its run never reaches mflux's generate and save."""
    out = tmp_path / "out.png"
    run_command(["--prompt", "x", "--seed", "1", "--steps", "4", "--output", str(out)], adapter=z_image_turbo.ADAPTER)
    (model,) = FakeZImage.instances
    assert model.model_config is AVAILABLE_MODELS["z-image-turbo"]
    assert out.is_file()


def test_turbo_warns_ignored_guidance_before_load(run_command, tmp_path: Path) -> None:
    """Bug: the Turbo adapter's pre-load hook drops mflux's warn_ignored_options(IGNORED_OPTIONS) call, or it runs after
    load(), so --guidance on Turbo goes unmentioned or is mentioned only after the weights load."""
    seen: list[tuple[type[Warning], str, int]] = []

    def record(message: Warning | str, category: type[Warning], *args: Any, **kwargs: Any) -> None:
        seen.append((category, str(message), len(FakeZImage.instances)))

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = record
        run_command(
            ["--prompt", "x", "--seed", "1", "--steps", "4", "--guidance", "3.5", "--output", str(tmp_path / "o.png")],
            adapter=z_image_turbo.ADAPTER,
        )
    assert [(category, models) for category, message, models in seen if "guidance-distilled" in message] == [
        (UserWarning, 0)
    ]
    assert len(FakeZImage.instances) == 1  # the run did load the model, after the warning


# A --model that is neither a model name, a path that exists nor the org/model shape of a Hub repo id.
NOT_FOUND = [
    "flux-dev",
    "./no-such-folder",
    "../no-such-folder",
    "~/no-such-folder",
    "/no/such/folder",
    "no/such/folder",
]


@pytest.mark.parametrize("model", NOT_FOUND)
def test_mflux_cannot_resolve_the_names_the_plugin_refuses(monkeypatch, tmp_path: Path, model) -> None:
    """Bug: mflux starts to resolve one of these names (a bare repo id, say), and the refusal below turns away a model
    that now loads. Red here means: change _refuse_unknown_model and this list."""
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="Model not found"):
        PathResolution.resolve(model)


@pytest.mark.parametrize(
    ("adapter", "command"),
    [
        pytest.param(z_image.ADAPTER, "mflux-generate-z-image-taef", id="base"),
        pytest.param(z_image_turbo.ADAPTER, "mflux-generate-z-image-turbo-taef", id="turbo"),
    ],
)
@pytest.mark.parametrize("model", NOT_FOUND)
def test_a_model_mflux_cannot_resolve_is_refused_before_load(
    run_command, capsys, monkeypatch, tmp_path: Path, model, adapter, command
) -> None:
    """Bug: run() leaves --model to load(), whose PathResolution raises FileNotFoundError outside run()'s try, so a
    typo ends in an uncaught traceback (exit 1), not in a refusal; or the refusal comes after the preview decoder was
    built (a download on first use)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(SystemExit) as exit_info:
        run_command(["--prompt", "x", "--model", model], adapter)
    assert exit_info.value.code == 2
    assert FakeZImage.instances == []
    assert FakeLivePreview.instances == []
    assert capsys.readouterr().err.splitlines()[-1] == (
        f"{command}: error: Model not found: {model!r}. "
        "Give a model name this command knows, a local path that exists or a Hugging Face repo id (org/model)."
    )


@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.parametrize(
    "model", ["weights", "./weights", "~/weights", "{home}/weights", "{home}/weights/", "one.safetensors"]
)
def test_a_local_path_that_exists_is_not_refused(run_command, monkeypatch, tmp_path: Path, model, adapter) -> None:
    """Bug: the check tests the shape of the name instead of the filesystem (or does not expand ~, or takes only
    folders), so a local checkpoint mflux loads is refused."""
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "weights").mkdir()
    (tmp_path / "one.safetensors").touch()
    model = model.format(home=tmp_path)
    run_command([*BASE, "--seed", "1", "--model", model], adapter)
    assert FakeZImage.instances[0].init_kwargs["model_path"] == model


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_a_repo_id_that_is_not_on_this_machine_is_not_refused(run_command, adapter) -> None:
    """Bug: the check asks only whether the name exists on disk, so every Hub repo id, which mflux downloads, is
    refused."""
    run_command([*BASE, "--seed", "1", "--model", "someorg/not-downloaded"], adapter)
    assert FakeZImage.instances[0].init_kwargs["model_path"] == "someorg/not-downloaded"


@pytest.mark.parametrize(
    ("adapter", "model"),
    [
        pytest.param(z_image.ADAPTER, "z-image", id="base"),
        pytest.param(z_image_turbo.ADAPTER, "z-image-turbo", id="turbo"),
    ],
)
def test_an_empty_model_name_runs_the_default_model(run_command, adapter, model) -> None:
    """Bug: the unknown-model refusal tests model_path against None only, so --model "" (which mflux's parser turns
    into model_path "" and its loader then runs as the command's default model) is refused as 'Model not found'."""
    run_command([*BASE, "--seed", "1", "--model", ""], adapter)
    (fake,) = FakeZImage.instances
    assert fake.model_config is AVAILABLE_MODELS[model]
    assert fake.init_kwargs["model_path"] == ""


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_a_name_is_left_to_mflux_when_its_path_rules_are_not_the_reviewed_ones(
    run_command, monkeypatch, tmp_path: Path, adapter
) -> None:
    """Bug: the refusal runs whatever mflux's model-path rules are, so on an mflux that adds a rule (a lookup of
    registry names, say) a --model that rule loads is refused before load. The reviewed rules refuse the same name:
    test_a_model_mflux_cannot_resolve_is_refused_before_load."""
    monkeypatch.setenv("HOME", str(tmp_path))
    added = Rule(priority=5, name="registry", check="is_registry_name", action=PathAction.LOCAL)
    monkeypatch.setattr(PathResolution, "RULES", PathResolution.RULES | {added})
    run_command([*BASE, "--seed", "1", "--model", "flux-dev"], adapter)
    (built,) = FakeZImage.instances
    assert built.init_kwargs["model_path"] == "flux-dev"


@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.parametrize("rules", [frozenset({object()}), None], ids=["rules-without-fields", "rules-not-a-set"])
def test_a_name_is_left_to_mflux_when_its_path_rules_cannot_be_read(
    run_command, monkeypatch, tmp_path: Path, adapter, rules
) -> None:
    """Bug: the rule-set comparison reads rule.priority and rule.check (or iterates RULES) unguarded, so on an mflux
    that reshapes its rules every unknown --model ends in an AttributeError or TypeError traceback before load, not in
    mflux's own handling of the name."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(PathResolution, "RULES", rules)
    run_command([*BASE, "--seed", "1", "--model", "flux-dev"], adapter)
    (built,) = FakeZImage.instances
    assert built.init_kwargs["model_path"] == "flux-dev"


def test_mflux_still_leaves_a_replayed_sidecar_open(monkeypatch, tmp_path: Path) -> None:
    """Bug: mflux closes the -C file now, and SIDECAR_LEFT_OPEN_BY_MFLUX stays on these tests, where it would hide a
    sidecar this package itself leaves open. Red here means: delete the filter (tests/_marks.py) and its uses."""
    sidecar = tmp_path / "replay.metadata.json"
    sidecar.write_text(json.dumps({"model": "Tongyi-MAI/Z-Image", "prompt": "p", "seed": 3, "steps": 5}))
    monkeypatch.setattr(sys, "argv", ["mflux-generate-z-image", "-C", str(sidecar)])
    with pytest.warns(ResourceWarning) as caught:
        z_image_generate.build_parser().parse_args()
        gc.collect()
    assert [str(w.message) for w in caught if re.match(SIDECAR_LEFT_OPEN_MESSAGE, str(w.message))] != []


# Each command with the repo id its own sidecars record as "model".
SIDECAR_MODELS = [
    pytest.param(z_image.ADAPTER, "mflux-generate-z-image-taef", "Tongyi-MAI/Z-Image", id="base"),
    pytest.param(z_image_turbo.ADAPTER, "mflux-generate-z-image-turbo-taef", "Tongyi-MAI/Z-Image-Turbo", id="turbo"),
]


def gone_sidecar(tmp_path: Path, model: str) -> Path:
    """A -C sidecar from a run on a local folder, ./gone, that no longer exists."""
    sidecar = tmp_path / "gone.metadata.json"
    sidecar.write_text(json.dumps({"model": model, "model_path": "./gone", "prompt": "p", "seed": 3, "steps": 5}))
    return sidecar


@SIDECAR_LEFT_OPEN_BY_MFLUX
@pytest.mark.parametrize(("adapter", "command", "model"), SIDECAR_MODELS)
def test_a_model_path_restored_from_a_sidecar_is_named_as_coming_from_it(
    run_command, capsys, tmp_path: Path, adapter, command, model
) -> None:
    """Bug: the refusal of a missing path never says the path came from the -C file, so the user looks for a --model
    they never typed."""
    sidecar = gone_sidecar(tmp_path, model)
    with pytest.raises(SystemExit) as exit_info:
        run_command(["-C", str(sidecar)], adapter)
    assert exit_info.value.code == 2
    assert FakeZImage.instances == []
    assert FakeLivePreview.instances == []
    assert capsys.readouterr().err.splitlines()[-2:] == [
        f"{command}: error: Model not found: './gone'. "
        "Give a model name this command knows, a local path that exists or a Hugging Face repo id (org/model).",
        f"Tip: that path came from {sidecar}. Pass --model PATH to use your own copy.",
    ]


@SIDECAR_LEFT_OPEN_BY_MFLUX
@pytest.mark.parametrize("typed", [["--model", "./gone"], ["-m", "./gone"], ["--model=./gone"], ["-m./gone"]])
def test_a_typed_model_path_is_not_named_as_coming_from_the_sidecar(run_command, capsys, tmp_path, typed) -> None:
    """Bug: the tip checks only that -C was given, so a --model (or -m) typed next to it is said to come from the -C
    file."""
    with pytest.raises(SystemExit) as exit_info:
        run_command(["-C", str(gone_sidecar(tmp_path, "Tongyi-MAI/Z-Image")), *typed])
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert "Model not found: './gone'." in err
    assert "Tip:" not in err


# The library wiring: the preview callback is built before load, registered ahead of mflux's callbacks and pointed at
# each seed's own file. FakeLivePreview stands in for mlx-taef's LivePreviewCallback (run_command swaps it in). Each
# test runs both commands.
PREVIEW_BASE = ["--prompt", "x", "--steps", "4"]


@pytest.fixture
def looping_model(run_command, monkeypatch) -> None:
    """Swaps in the fake that runs mflux's denoising loop for both commands, so the preview's callbacks fire."""
    monkeypatch.setattr(z_image_turbo_generate, "ZImage", LoopingFakeZImage)
    monkeypatch.setattr(z_image_generate, "ZImage", LoopingFakeZImage)


def before_paths() -> list[object]:
    (preview,) = FakeLivePreview.instances
    return [value for kind, value in preview.events if kind == "before"]


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_preview_is_built_before_the_model_loads_without_the_model(run_command, adapter) -> None:
    """Bug: run() builds the callback after load() (so a missing decoder is found only after the 6-20 GB model
    loads), passes flux=model, or leaves one of the five keywords at the library's default (every=5, previews switched
    off on error left to "raise")."""
    run_command([*PREVIEW_BASE, "--seed", "1"], adapter)
    (preview,) = FakeLivePreview.instances
    assert preview.models_at_init == 0
    assert preview.kwargs == {
        "variant": "zimage",
        "every": 1,
        "save_to": Path("image.preview.png"),
        "numbered_frames": False,
        "on_error": "disable",
    }


@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.parametrize(
    ("argv", "every", "frames"),
    [
        pytest.param(["--taef-frames"], 1, True, id="frames"),
        pytest.param(["--taef-every", "3"], 3, False, id="every-3"),
    ],
)
def test_frames_and_every_reach_the_callback(run_command, argv, every, frames, adapter) -> None:
    """Bug: --taef-every or --taef-frames is parsed and checked but never handed to the callback, so the preview
    keeps the default cadence or overwrites one file."""
    run_command([*PREVIEW_BASE, "--seed", "1", *argv], adapter)
    (preview,) = FakeLivePreview.instances
    assert (preview.kwargs["every"], preview.kwargs["numbered_frames"]) == (every, frames)


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_preview_runs_before_mflux_callbacks(run_command, adapter) -> None:
    """Bug: the preview is registered after CallbackManager.register_callbacks (or not at all), so mflux's MemorySaver
    runs first in each step and the battery check can stop a run before the preview's per-seed reset."""
    run_command([*PREVIEW_BASE, "--seed", "1"], adapter)
    (preview,) = FakeLivePreview.instances
    registry = FakeZImage.instances[0].callbacks
    assert registry.before_loop[0] is preview
    assert registry.in_loop[0] is preview
    assert [type(c).__name__ for c in registry.in_loop[1:]] == ["MemorySaver"]


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_each_seed_gets_its_own_path_before_its_generation(run_command, looping_model, adapter) -> None:
    """Bug: save_to is set once for the first seed (seed 2 overwrites seed 1's preview), or set to a str, which the
    library's numbered-frame names (save_to.with_name) cannot use and on_error="disable" hides."""
    run_command([*PREVIEW_BASE, "--seed", "1", "2"], adapter)
    paths = before_paths()
    assert paths == [Path("image_seed_1.preview.png"), Path("image_seed_2.preview.png")]
    assert all(isinstance(path, Path) for path in paths)


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_explicit_path_is_one_file_across_seeds(run_command, looping_model, adapter) -> None:
    """Bug: run() adds a seed suffix to an explicit path without frames, so a viewer open on live.png stops
    updating after the first seed."""
    run_command([*PREVIEW_BASE, "--seed", "1", "2", "--taef-preview", "live.png"], adapter)
    assert before_paths() == [Path("live.png"), Path("live.png")]


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_start_line_names_the_path_cadence_and_decoder(run_command, capsys, adapter) -> None:
    """Bug: the start line is missing, printed once for the whole run, or printed after the generation, so a user
    does not know where to point a viewer until the image is done."""
    run_command([*PREVIEW_BASE, "--seed", "1"], adapter)
    out = capsys.readouterr().out
    assert out.count("TAEF preview: image.preview.png, every step (zimage decoder)\n") == 1
    (model,) = FakeZImage.instances
    assert len(model.generate_calls) == 1


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_start_line_comes_before_each_generation(run_command, monkeypatch, capsys, adapter) -> None:
    """Bug: run() prints the start line after generate() returns, so the line names the preview only once the image
    it previews is done."""
    printed_at_generate: list[str] = []
    original = FakeZImage.generate_image

    def generate_image(self, **kwargs):
        printed_at_generate.append(capsys.readouterr().out)
        return original(self, **kwargs)

    monkeypatch.setattr(FakeZImage, "generate_image", generate_image)
    run_command([*PREVIEW_BASE, "--seed", "1", "2"], adapter)
    assert ["TAEF preview: image_seed_1.preview.png" in out for out in printed_at_generate] == [True, False]
    assert ["TAEF preview: image_seed_2.preview.png" in out for out in printed_at_generate] == [False, True]


@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.parametrize(
    ("argv", "lines"),
    [
        pytest.param(
            ["--seed", "1", "2"],
            [
                "TAEF preview: image_seed_1.preview.png, every step (zimage decoder)",
                "TAEF preview: image_seed_2.preview.png, every step (zimage decoder)",
            ],
            id="default",
        ),
        pytest.param(
            ["--seed", "1", "2", "--taef-preview", "live.png"],
            ["TAEF preview: live.png, every step (zimage decoder)"] * 2,
            id="one-live-file",
        ),
        pytest.param(
            ["--seed", "1", "2", "--taef-frames", "--taef-preview", "live.png"],
            [
                "TAEF preview frames: live_seed_1_stepNN.png, every step (zimage decoder)",
                "TAEF preview frames: live_seed_2_stepNN.png, every step (zimage decoder)",
            ],
            id="frames",
        ),
        # The line names the folder the frames go to, and no file frames mode never writes.
        pytest.param(
            ["--seed", "1", "--taef-frames", "--taef-preview", "shots/live.png"],
            ["TAEF preview frames: shots/live_stepNN.png, every step (zimage decoder)"],
            id="frames-in-a-folder",
        ),
        pytest.param(
            ["--seed", "1", "--taef-every", "3"],
            ["TAEF preview: image.preview.png, every 3 steps (zimage decoder)"],
            id="every-3",
        ),
    ],
)
def test_start_line_prints_the_real_path_per_seed(run_command, capsys, argv, lines, adapter) -> None:
    """Bug: the line prints the path as typed (or the first seed's path), not the one this seed writes, or the
    cadence it names is not the one the callback got."""
    run_command([*PREVIEW_BASE, *argv], adapter)
    printed = [line for line in capsys.readouterr().out.splitlines() if line.startswith("TAEF preview")]
    assert printed == lines


@pytest.mark.parametrize(("adapter", "command", "plain"), COMMANDS)
def test_decoder_failure_exits_1_before_load(run_command, capsys, adapter, command, plain) -> None:
    """Bug: a decoder that cannot be prepared (no network, empty cache) escapes as a traceback, exits 2 like a
    refusal, is found only after the model has loaded, or the line names the other command."""
    FakeLivePreview.fail_on_init.append(RuntimeError("offline"))
    with pytest.raises(SystemExit) as exit_info:
        run_command([*PREVIEW_BASE, "--seed", "1"], adapter)
    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert (
        f"{command}: error: could not prepare the TAEF preview decoder: offline. "
        f"Run {plain} to generate without previews." in err
    )
    assert FakeZImage.instances == []


def _keys(value: object) -> list[str]:
    if isinstance(value, dict):
        return [key for key, item in value.items() for key in (key, *_keys(item))]
    if isinstance(value, list):
        return [key for item in value for key in _keys(item)]
    return []


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_the_image_record_is_the_plain_commands(run_command, monkeypatch, tmp_path: Path, adapter) -> None:
    """Bug: the plugin writes its preview settings into the image's record (generation_parameters or the sidecar),
    so a -C replay of the image with plain mflux carries keys mflux does not know."""
    saved: list[object] = []
    original = GeneratedImage.save

    def save(self, *args, **kwargs):
        saved.append(self.generation_parameters)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GeneratedImage, "save", save)
    out = tmp_path / "out.png"
    run_command([*PREVIEW_BASE, "--seed", "1", "--make-conf", "--output", str(out)], adapter)
    sidecar = json.loads(out.with_suffix(".metadata.json").read_text())
    assert sidecar  # the record was written
    assert [key for key in _keys(sidecar) if "taef" in key.lower()] == []
    (model,) = FakeZImage.instances
    assert len(saved) == 1
    assert saved[0] is model.returned_parameters[0]


@pytest.mark.parametrize(("adapter", "command", "plain"), COMMANDS)
def test_failure_after_load_exits_1_with_the_plain_hint(run_command, capsys, adapter, command, plain) -> None:
    """Bug: a failure after load names the wrong command or the wrong plain command (or none) in its hint."""
    FakeZImage.fail_on_seed[1] = RuntimeError("boom")
    with pytest.raises(SystemExit) as exit_info:
        run_command([*PREVIEW_BASE, "--seed", "1"], adapter)
    assert exit_info.value.code == 1
    assert (
        f"{command}: error: the run failed (traceback above). Run {plain} to generate without previews."
        in capsys.readouterr().err
    )


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_ctrl_c_exits_0_and_saves_nothing(run_command, looping_model, capsys, tmp_path: Path, adapter) -> None:
    """Bug: Ctrl-C in the denoising loop becomes a traceback or exit 1 now that a preview is registered, or an image
    is saved for the stopped seed."""
    LoopingFakeZImage.interrupt_at_step = 2
    run_command([*PREVIEW_BASE, "--seed", "1"], adapter)
    assert "Stopping image generation at step 3/4" in capsys.readouterr().out
    assert not (tmp_path / "image.png").exists()
    (preview,) = FakeLivePreview.instances
    assert [event for event in preview.events if event[0] == "in"] == [("in", 0), ("in", 1)]


# Paths only the base command has: it also runs Z-Image Turbo, takes a third-party copy of Z-Image through
# --base-model, and keeps mflux's negative-prompt warning.
def test_base_command_runs_turbo_through_its_family(run_command) -> None:
    """Bug: the base command stops honoring mflux's family (it refuses --model z-image-turbo, or loads the base
    z-image config for it), so a Turbo run through the base command fails or runs the wrong model."""
    run_command([*PREVIEW_BASE, "--seed", "1", "--model", "z-image-turbo"], z_image.ADAPTER)
    (model,) = FakeZImage.instances
    assert model.model_config is AVAILABLE_MODELS["z-image-turbo"]
    (preview,) = FakeLivePreview.instances
    assert preview.variant == "zimage"


def test_base_copy_with_base_model_previews_with_zimage(run_command) -> None:
    """Bug: the decoder is chosen from the --model name rather than the model config mflux resolved, so a quantized
    community copy of Z-Image named only through --base-model is refused as a model TAEF has no decoder for, or the
    preview is built after the model loads."""
    run_command(
        [*PREVIEW_BASE, "--seed", "1", "--model", "mflux-community/z-image-base-mflux-q4", "--base-model", "z-image"],
        z_image.ADAPTER,
    )
    (model,) = FakeZImage.instances
    assert model.model_config.model_name == "Tongyi-MAI/Z-Image"
    (preview,) = FakeLivePreview.instances
    assert preview.variant == "zimage"
    assert preview.models_at_init == 0


def test_base_negative_prompt_warning_still_comes_before_load(run_command) -> None:
    """Bug: the base adapter's pre-load hook drops mflux's negative-prompt warning, or it runs after load(), so a
    --negative-prompt that has no effect at the default guidance is mentioned only after the weights load, or never."""
    seen: list[tuple[type[Warning], str, int]] = []

    def record(message: Warning | str, category: type[Warning], *args: Any, **kwargs: Any) -> None:
        seen.append((category, str(message), len(FakeZImage.instances)))

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = record
        run_command([*PREVIEW_BASE, "--seed", "1", "--negative-prompt", "x"], z_image.ADAPTER)
    assert [
        (category, models) for category, message, models in seen if "--negative-prompt has no effect" in message
    ] == [(UserWarning, 0)]
    assert len(FakeZImage.instances) == 1  # the run did load the model, after the warning
