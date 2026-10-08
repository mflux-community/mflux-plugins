"""The plugin command runs mflux's own Z-Image steps, with TeaCache applied in between."""

import dataclasses
from pathlib import Path

import PIL.Image
import pytest
from _fakes import FakeZImage
from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.models.common.config.model_config import AVAILABLE_MODELS

from mflux.extras.teacache import z_image


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


def test_every_flag_reaches_mflux_steps_unchanged(run_command, teacache, tmp_path, ref_png, lora_file) -> None:
    """Bug: _core builds its own model or generate call instead of calling mflux's load()/generate(),
    a flag (LoRA scale, img2img strength, PiD, size, --float32) is dropped or rewritten on the way, or
    TeaCache is applied once per seed instead of once per run."""
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
        ]
    )  # fmt: skip
    (model,) = FakeZImage.instances
    assert len(teacache.calls) == 1
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


@pytest.mark.parametrize(("argv", "expected"), [([], None), (["--teacache-threshold", "0.3"], 0.3)])
def test_the_threshold_reaches_apply_teacache(run_command, teacache, argv, expected) -> None:
    """Bug: the plugin passes its own default instead of None (so mlx-teacache's per-model default is
    never used), or drops the user's --teacache-threshold."""
    run_command(["--prompt", "x", "--seed", "1", "--steps", "5", *argv])
    assert teacache.calls == [{"rel_l1_thresh": expected}]


@pytest.mark.parametrize(
    ("argv", "expected"), [([], "flow_match_euler_discrete"), (["--scheduler", "linear"], "linear")]
)
def test_the_scheduler_default_applies_only_when_the_flag_is_absent(run_command, teacache, argv, expected) -> None:
    """Bug: the plugin builds its own parser (or edits mflux's) so mflux's scheduler default never reaches generate(),
    or run() overrides an explicit --scheduler."""
    run_command(["--prompt", "x", "--seed", "1", "--steps", "5", *argv])
    assert FakeZImage.instances[0].generate_calls[0]["scheduler"] == expected


def test_mflux_own_option_warnings_still_fire(run_command, teacache) -> None:
    """Bug: run() never calls the after_checks hook, so a negative prompt that has no effect at the
    default guidance goes unmentioned, unlike in mflux-generate-z-image."""
    with pytest.warns(UserWarning, match="--negative-prompt has no effect"):
        run_command(["--prompt", "x", "--seed", "1", "--steps", "5", "--negative-prompt", "blurry"])


def test_before_validate_runs_after_parsing_and_before_validate(run_command, teacache, capsys) -> None:
    """Bug: run() never calls the before_validate hook (or calls it after validate), so an adapter whose mflux command
    adjusts the parsed arguments before validate() validates and loads the unadjusted ones."""
    validated: list[str | None] = []

    def validate(args):
        validated.append(args.model)
        return z_image.ADAPTER.validate(args)

    hooked = dataclasses.replace(
        z_image.ADAPTER, before_validate=lambda args: setattr(args, "model", "dev"), validate=validate
    )
    with pytest.raises(SystemExit) as exc:
        run_command(["--prompt", "x"], hooked)
    assert exc.value.code == 2
    assert "'dev' is not Tongyi-MAI/Z-Image" in capsys.readouterr().err
    # mflux's load() validates again, so the error alone can't show the order; what validate saw can.
    assert validated == ["dev"]
    assert FakeZImage.instances == []


def test_an_adapter_without_optional_hooks_runs(run_command, teacache) -> None:
    """Bug: run() calls a hook unconditionally, so a future adapter whose mflux command has no pre-load
    block must pass stub functions, or crashes with 'NoneType' object is not callable."""
    bare = dataclasses.replace(z_image.ADAPTER, before_validate=None, after_checks=None, active_steps=None)
    run_command(["--prompt", "x", "--seed", "1", "--steps", "5", "--scheduler", "linear"], bare)
    assert len(FakeZImage.instances[0].generate_calls) == 1


def test_teacache_is_applied_before_mflux_registers_its_callbacks(run_command, teacache) -> None:
    """Bug: run() registers mflux's callbacks before apply_teacache, so MemorySaver's after-loop step
    (mx.clear_cache, or dropping the transformer with --low-ram) runs before TeaCache's own after-loop
    callback releases its cached residuals; those buffers then go back to MLX's cache pool after the
    clear and stay resident through the decode."""
    run_command(["--prompt", "x", "--seed", "1", "--steps", "5"])
    after_loop = FakeZImage.instances[0].callbacks.after_loop
    kinds = [type(callback).__name__ for callback in after_loop]
    assert kinds.index("LifecycleMarker") < kinds.index("MemorySaver")
    assert isinstance(after_loop[-1], MemorySaver)
