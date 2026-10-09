"""The template command runs mflux's own Z-Image steps, with the plugin's hooks in between."""

import dataclasses
import json
import warnings
from pathlib import Path
from typing import Any

import PIL.Image
import pytest
from _fakes import FakeZImage
from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.utils.exceptions import ModelConfigError, StopImageGenerationException

from mflux.extras.myplugin import _core, z_image

BASE = ["--prompt", "a puffin", "--steps", "4"]


class Recorder:
    """Hooks for an adapter that record what run() hands them. apply returns HANDLE, before_generate returns a
    per-seed state, so a test can see both reach after_generate."""

    HANDLE = object()

    def __init__(self) -> None:
        self.applied: list[tuple[Any, float | None, int, int]] = []
        self.before: list[tuple[Any, Any, int, int]] = []
        self.after: list[tuple[Any, Any, int]] = []

    def apply(self, model: Any, args: Any) -> object:
        self.applied.append((model, args.myplugin_option, len(model.generate_calls), len(model.callbacks.after_loop)))
        return self.HANDLE

    def before_generate(self, model: Any, args: Any, handle: Any, seed: int) -> str:
        self.before.append((model, handle, seed, len(model.generate_calls)))
        return f"state-{seed}"

    def after_generate(self, image: Any, args: Any, handle: Any, state: Any) -> None:
        self.after.append((handle, state, image.seed))

    def adapter(self) -> _core.Adapter:
        return dataclasses.replace(
            z_image.ADAPTER, apply=self.apply, before_generate=self.before_generate, after_generate=self.after_generate
        )


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
    scale, img2img strength, PiD, size, --float32) is dropped or rewritten on the way, or apply runs once per seed or
    sees an option value the user never gave."""
    recorder = Recorder()
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
        recorder.adapter(),
    )  # fmt: skip
    (model,) = FakeZImage.instances
    assert recorder.applied == [(model, None, 0, 0)]
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


def test_without_the_option_mflux_registers_its_usual_callbacks(run_command) -> None:
    """Bug: the template registers a callback of its own, or skips CallbackManager.register_callbacks, so a run without
    --myplugin-option differs from mflux-generate-z-image, which registers BatterySaver and MemorySaver (mflux 0.22,
    no --stepwise-image-output-dir)."""
    run_command([*BASE, "--seed", "5"])
    registry = FakeZImage.instances[0].callbacks
    assert [type(c).__name__ for c in registry.before_loop] == ["BatterySaver", "MemorySaver"]
    assert [type(c).__name__ for c in registry.in_loop] == ["MemorySaver"]
    assert [type(c).__name__ for c in registry.after_loop] == ["MemorySaver"]
    assert registry.interrupt == []


def test_the_option_reaches_apply_once_before_the_first_image(run_command) -> None:
    """Bug: apply reads --myplugin-option from somewhere other than the parsed arguments (a default, or a value set
    before parsing), so a library would attach with the wrong setting. (Once, before the first image, is also covered
    by test_every_flag_reaches_mflux_steps_unchanged; this test is the one that passes a value.)"""
    recorder = Recorder()
    run_command([*BASE, "--seed", "1", "2", "--myplugin-option", "0.5"], recorder.adapter())
    (model,) = FakeZImage.instances
    assert recorder.applied == [(model, 0.5, 0, 0)]
    assert [call["seed"] for call in model.generate_calls] == [1, 2]


def test_the_handle_and_the_per_seed_state_reach_after_generate(run_command) -> None:
    """Bug: run() drops what apply returned, calls before_generate after generate (or once per run), or hands
    after_generate another seed's state, so a library can't attach its record to the right image."""
    recorder = Recorder()
    run_command([*BASE, "--seed", "1", "2"], recorder.adapter())
    (model,) = FakeZImage.instances
    handle = Recorder.HANDLE
    assert recorder.before == [(model, handle, 1, 0), (model, handle, 2, 1)]
    assert recorder.after == [(handle, "state-1", 1), (handle, "state-2", 2)]


def test_apply_runs_before_mflux_registers_its_callbacks(run_command) -> None:
    """Bug: run() registers mflux's callbacks before apply, so MemorySaver's after-loop step (mx.clear_cache, or
    dropping the transformer with --low-ram) runs before a callback the library registers in apply; buffers that
    callback frees then stay in MLX's cache pool through the decode."""

    class LibraryCallback:
        def call_after_loop(self, **kwargs: Any) -> None:
            pass

    def apply(model: Any, args: Any) -> None:
        model.callbacks.register(LibraryCallback())

    run_command([*BASE, "--seed", "1"], dataclasses.replace(z_image.ADAPTER, apply=apply))
    after_loop = FakeZImage.instances[0].callbacks.after_loop
    kinds = [type(callback).__name__ for callback in after_loop]
    assert kinds.index("LibraryCallback") < kinds.index("MemorySaver")
    assert isinstance(after_loop[-1], MemorySaver)


def test_one_image_is_saved_per_seed(run_command, tmp_path: Path) -> None:
    """Bug: the loop saves only the last seed, or ignores --output's {seed} field. With more than one seed, mflux's
    parse_args itself appends _seed_{seed} to the output stem, hence out_3_seed_3.png."""
    run_command([*BASE, "--seed", "3", "4", "--output", "out_{seed}.png"])
    assert sorted(p.name for p in tmp_path.glob("out_*.png")) == ["out_3_seed_3.png", "out_4_seed_4.png"]


@pytest.mark.parametrize("value", ["0", "-1", "nan"])
def test_a_refused_option_exits_2_before_any_weights_load(run_command, capsys, value: str) -> None:
    """Bug: refuse runs after load (weights already read), lets NaN through, or exits with a code other than
    argparse's 2."""
    with pytest.raises(SystemExit) as exit_info:
        run_command([*BASE, "--myplugin-option", value])
    assert exit_info.value.code == 2
    assert FakeZImage.instances == []
    assert (
        f"--myplugin-option must be greater than 0, got {float(value)}. Run mflux-generate-z-image instead."
        in capsys.readouterr().err
    )


def test_a_refused_run_prints_no_mflux_option_warnings(run_command, capsys) -> None:
    """Bug: run() calls after_checks before refuse, so a refused run first warns that --negative-prompt has no effect
    instead of printing one clear refusal."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(SystemExit) as exit_info:
            run_command([*BASE, "--negative-prompt", "blurry", "--myplugin-option", "0"])
    assert exit_info.value.code == 2
    assert "--myplugin-option must be greater than 0" in capsys.readouterr().err


def test_mflux_own_option_warnings_still_fire(run_command) -> None:
    """Bug: run() never calls the after_checks hook, so a negative prompt that has no effect at the default guidance
    goes unmentioned, unlike in mflux-generate-z-image."""
    with pytest.warns(UserWarning, match="--negative-prompt has no effect"):
        run_command([*BASE, "--seed", "1", "--negative-prompt", "blurry"])


def test_an_adapter_without_optional_hooks_runs(run_command) -> None:
    """Bug: run() calls an optional hook unconditionally, so an adapter whose mflux command needs no refusal or
    pre-load warning must pass stub functions, or crashes with 'NoneType' object is not callable."""
    bare = dataclasses.replace(
        z_image.ADAPTER, refuse=None, after_checks=None, before_generate=None, after_generate=None
    )
    run_command([*BASE, "--seed", "1", "--myplugin-option", "0"], bare)  # 0 runs: no refuse hook
    assert len(FakeZImage.instances[0].generate_calls) == 1


def test_a_model_config_error_becomes_exit_2(run_command, capsys) -> None:
    """Bug: a ModelConfigError from validate escapes as a traceback instead of mflux's usage error."""

    def failing_validate(args):
        raise ModelConfigError("no such model")

    with pytest.raises(SystemExit) as exit_info:
        run_command(BASE, dataclasses.replace(z_image.ADAPTER, validate=failing_validate))
    assert exit_info.value.code == 2
    assert "no such model" in capsys.readouterr().err
    assert FakeZImage.instances == []


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
        "mflux-generate-z-image-myplugin: error: the run failed (traceback above). "
        "Run mflux-generate-z-image to generate without the plugin." in captured.err
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


def record_option(image: Any, args: Any, handle: Any, state: Any) -> None:
    image.generation_parameters = {**image.generation_parameters, "myplugin_option": args.myplugin_option}


def test_after_generate_records_into_the_saved_sidecar(run_command, tmp_path: Path) -> None:
    """Bug: run() calls after_generate after save, so the plugin's record never reaches the .metadata.json file."""
    adapter = dataclasses.replace(z_image.ADAPTER, after_generate=record_option)
    run_command([*BASE, "--seed", "6", "--myplugin-option", "0.25", "--metadata", "--output", "img.png"], adapter)
    sidecar = json.loads((tmp_path / "img.metadata.json").read_text(encoding="utf-8"))
    assert sidecar["myplugin_option"] == 0.25
    assert sidecar["seed"] == 6


def test_a_run_without_metadata_writes_no_sidecar(run_command, tmp_path: Path) -> None:
    """Bug: run() saves with export_json_metadata=True (or ignores --metadata), so every image leaves a
    .metadata.json next to it, unlike mflux-generate-z-image."""
    adapter = dataclasses.replace(z_image.ADAPTER, after_generate=record_option)
    run_command([*BASE, "--seed", "6", "--myplugin-option", "0.25", "--output", "img.png"], adapter)
    assert (tmp_path / "img.png").is_file()
    assert list(tmp_path.glob("*.metadata.json")) == []
