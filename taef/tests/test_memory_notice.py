"""At 1024x1024 and above, without --low-ram or --mlx-cache-limit-gb, each command prints one memory notice on stderr
before the model loads. The expected lines are spelled out here, not built from the code."""

import dataclasses
import json
from pathlib import Path
from typing import Any

import PIL.Image
import pytest
from _fakes import FakeLivePreview, FakeZImage
from _marks import SIDECAR_LEFT_OPEN_BY_MFLUX

from mflux.extras.taef import z_image, z_image_turbo

BASE = ["--prompt", "a puffin", "--steps", "4"]
# (adapter, its own command), spelled out per command.
COMMANDS = [
    pytest.param(z_image_turbo.ADAPTER, "mflux-generate-z-image-turbo-taef", id="turbo"),
    pytest.param(z_image.ADAPTER, "mflux-generate-z-image-taef", id="base"),
]


def notice(command: str, size: str) -> str:
    return (
        f"{command}: note: at {size} the live preview uses extra memory. --low-ram or --mlx-cache-limit-gb 1 lowers "
        'that extra memory (see "Memory at 1024×1024" in the mflux-taef README).'
    )


def notes(err: str, command: str) -> list[str]:
    """Every line of `err` that starts like a notice of this command."""
    return [line for line in err.splitlines() if line.startswith(f"{command}: note:")]


@pytest.mark.parametrize(("adapter", "command"), COMMANDS)
def test_the_notice_prints_at_the_default_size_without_a_flag(run_command, capsys, adapter, command) -> None:
    """Bug: the size check reads only a typed --width/--height (or the Turbo command's "auto" default as no size), so
    the most common run, at mflux's 1024x1024 default, never sees the notice."""
    run_command([*BASE, "--seed", "1"], adapter)
    assert notes(capsys.readouterr().err, command) == [notice(command, "1024x1024")]
    assert len(FakeZImage.instances) == 1


@pytest.mark.parametrize(("adapter", "command"), COMMANDS)
@pytest.mark.parametrize("flag", [["--low-ram"], ["--mlx-cache-limit-gb", "1"]], ids=["low-ram", "cache-limit"])
def test_a_memory_flag_silences_the_notice(run_command, capsys, adapter, command, flag) -> None:
    """Bug: the notice ignores --low-ram (or --mlx-cache-limit-gb), so a user who already followed its advice is told
    to add the flag they passed."""
    run_command([*BASE, "--seed", "1", "--width", "1024", "--height", "1024", *flag], adapter)
    assert notes(capsys.readouterr().err, command) == []
    assert len(FakeZImage.instances) == 1


@pytest.mark.parametrize(("adapter", "command"), COMMANDS)
@pytest.mark.parametrize(
    ("width", "height", "printed"),
    [
        pytest.param("1024", "1008", False, id="one-row-of-16-under"),
        pytest.param("1024", "1024", True, id="exactly-1024x1024"),
        pytest.param("1536", "768", True, id="not-square-above"),
    ],
)
def test_the_notice_starts_at_1024_by_1024_pixels(
    run_command, capsys, adapter, command, width, height, printed
) -> None:
    """Bug: the threshold compares with > instead of >= (silent at exactly 1024x1024), compares each side with 1024
    instead of the pixel count (silent at 1536x768), or prints below the size (at 1024x1008)."""
    run_command([*BASE, "--seed", "1", "--width", width, "--height", height], adapter)
    expected = [notice(command, f"{width}x{height}")] if printed else []
    assert notes(capsys.readouterr().err, command) == expected


@pytest.mark.parametrize(("adapter", "command"), COMMANDS)
@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        # mflux's Config runs 1040x1015 at 1040x1008: 1,048,320 pixels, under 1024x1024.
        pytest.param("1040", "1015", None, id="rounds-down-under"),
        # 1030x1030 runs at 1024x1024.
        pytest.param("1030", "1030", "1024x1024", id="rounds-down-to-1024"),
        # 1020x1020 runs at 1008x1008 (mflux floors to 16); rounding to the nearest 16 would say 1024x1024.
        pytest.param("1020", "1020", None, id="floors-not-nearest"),
    ],
)
def test_the_notice_uses_the_size_mflux_rounds_to(
    run_command, capsys, adapter, command, width, height, expected
) -> None:
    """Bug: the notice checks and prints the typed size, not the multiple of 16 below it that mflux generates at, so
    1040x1015 gets a note for a run under 1024x1024 and 1030x1030 is reported at a size that never runs."""
    run_command([*BASE, "--seed", "1", "--width", width, "--height", height], adapter)
    assert notes(capsys.readouterr().err, command) == ([] if expected is None else [notice(command, expected)])


@pytest.mark.parametrize(("adapter", "command"), COMMANDS)
def test_several_seeds_print_the_notice_once(run_command, capsys, adapter, command) -> None:
    """Bug: the notice is printed inside the per-seed loop, so a run with two seeds prints it twice."""
    run_command([*BASE, "--seed", "1", "2"], adapter)
    assert notes(capsys.readouterr().err, command) == [notice(command, "1024x1024")]
    assert len(FakeZImage.instances[0].generate_calls) == 2


@pytest.mark.parametrize(("adapter", "command"), COMMANDS)
def test_the_notice_comes_before_the_model_loads(run_command, capsys, adapter, command) -> None:
    """Bug: the notice is printed after load() (or after the first generation), so the user reads the advice only
    once the weights already hold the memory."""
    err_at_load: list[str] = []

    def load(args: Any) -> Any:
        err_at_load.append(capsys.readouterr().err)
        return adapter.load(args)

    run_command([*BASE, "--seed", "1"], dataclasses.replace(adapter, load=load))
    assert len(err_at_load) == 1
    assert notes(err_at_load[0], command) == [notice(command, "1024x1024")]
    assert notes(capsys.readouterr().err, command) == []


@pytest.mark.parametrize(("adapter", "command"), COMMANDS)
def test_a_refused_run_prints_no_notice(run_command, capsys, adapter, command) -> None:
    """Bug: the notice is printed before the plugin's refusals, so a run that is refused (exit 2) at 1024x1024 prints
    a memory note next to the one error the user must fix."""
    with pytest.raises(SystemExit) as exit_info:
        run_command([*BASE, "--seed", "1", "--taef-every", "0"], adapter)
    assert exit_info.value.code == 2
    assert notes(capsys.readouterr().err, command) == []


@pytest.mark.parametrize(("adapter", "command"), COMMANDS)
def test_a_run_refused_for_an_unknown_model_prints_no_notice(run_command, capsys, adapter, command) -> None:
    """Bug: the notice is printed before the later refusals (mflux's validate, the unknown-model check), so a run
    refused with exit 2 for --model flux-dev also prints a memory note."""
    with pytest.raises(SystemExit) as exit_info:
        run_command([*BASE, "--seed", "1", "--model", "flux-dev"], adapter)
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert "Model not found: 'flux-dev'." in err
    assert notes(err, command) == []


@pytest.mark.parametrize(("adapter", "command"), COMMANDS)
def test_a_decoder_failure_prints_no_notice(run_command, capsys, adapter, command) -> None:
    """Bug: the notice is printed before the preview decoder is prepared, so a run that ends with exit 1 because the
    decoder is missing also prints memory advice for a preview that never runs."""
    FakeLivePreview.fail_on_init.append(RuntimeError("offline"))
    with pytest.raises(SystemExit) as exit_info:
        run_command([*BASE, "--seed", "1"], adapter)
    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert "could not prepare the TAEF preview decoder: offline." in err
    assert notes(err, command) == []


@pytest.fixture
def small_png(tmp_path: Path) -> Path:
    path = tmp_path / "small.png"
    PIL.Image.new("RGB", (64, 32)).save(path)
    return path


def test_turbo_with_an_init_image_and_no_size_prints_no_notice(run_command, capsys, small_png: Path) -> None:
    """Bug: the Turbo command's "auto" size is taken as mflux's 1024x1024 default even with --image, where mflux sizes
    the run from the image (64x32 here), so a small image-to-image run is told it is at 1024x1024."""
    run_command([*BASE, "--seed", "1", "--image", str(small_png), "0.5"], z_image_turbo.ADAPTER)
    assert notes(capsys.readouterr().err, "mflux-generate-z-image-turbo-taef") == []
    assert FakeZImage.instances[0].generate_calls[0]["width"] == 64


def test_turbo_with_an_init_image_and_a_typed_size_prints_the_notice(run_command, capsys, small_png: Path) -> None:
    """Bug: any --image silences the notice, although a typed --width and --height fix the size before load."""
    run_command(
        [*BASE, "--seed", "1", "--image", str(small_png), "0.5", "--width", "1024", "--height", "1024"],
        z_image_turbo.ADAPTER,
    )
    command = "mflux-generate-z-image-turbo-taef"
    assert notes(capsys.readouterr().err, command) == [notice(command, "1024x1024")]


@pytest.fixture
def large_png(tmp_path: Path) -> Path:
    path = tmp_path / "large.png"
    PIL.Image.new("RGB", (2048, 2048)).save(path)
    return path


def test_turbo_with_a_large_init_image_and_no_size_prints_the_image_size(run_command, capsys, large_png: Path) -> None:
    """Bug: the notice treats the Turbo command's "auto" size next to --image as unknown, so a 2048x2048
    image-to-image run, which mflux sizes from the image, gets no note."""
    run_command([*BASE, "--seed", "1", "--image", str(large_png), "0.5"], z_image_turbo.ADAPTER)
    command = "mflux-generate-z-image-turbo-taef"
    assert notes(capsys.readouterr().err, command) == [notice(command, "2048x2048")]
    assert FakeZImage.instances[0].generate_calls[0]["width"] == 2048


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        # mflux's DimensionResolver keeps the typed side and takes the "auto" side from the 2048x2048 image.
        pytest.param(["--width", "1024"], "1024x2048", id="width-typed"),
        pytest.param(["--height", "512"], "2048x512", id="height-typed"),
    ],
)
def test_turbo_with_an_init_image_and_one_side_typed_prints_the_mixed_size(
    run_command, capsys, large_png: Path, typed, expected
) -> None:
    """Bug: one typed side next to an "auto" side with --image counts as an unknown size (or takes the typed side for
    both), so the note is missing or names a size mflux does not run."""
    run_command([*BASE, "--seed", "1", "--image", str(large_png), "0.5", *typed], z_image_turbo.ADAPTER)
    command = "mflux-generate-z-image-turbo-taef"
    assert notes(capsys.readouterr().err, command) == [notice(command, expected)]


def test_turbo_with_a_scale_factor_and_no_image_prints_the_default_size(run_command, capsys) -> None:
    """Bug: a scale factor such as --width 2x without --image is read as an unknown size, or scaled from 1024, though
    mflux's DimensionResolver gives a scale factor without a reference image its default side, 1024."""
    run_command([*BASE, "--seed", "1", "--width", "2x"], z_image_turbo.ADAPTER)
    command = "mflux-generate-z-image-turbo-taef"
    assert notes(capsys.readouterr().err, command) == [notice(command, "1024x1024")]
    assert FakeZImage.instances[0].generate_calls[0]["width"] == 1024


def test_an_unreadable_init_image_leaves_the_failure_to_mflux(run_command, capsys, tmp_path: Path) -> None:
    """Bug: the notice reads the init image's header outside a guard, so a file that is no image ends the run inside
    the notice instead of in mflux's own image loading."""
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    command = "mflux-generate-z-image-turbo-taef"
    with pytest.raises(SystemExit) as exit_info:
        run_command([*BASE, "--seed", "1", "--image", str(broken), "0.5"], z_image_turbo.ADAPTER)
    assert exit_info.value.code == 1
    assert len(FakeZImage.instances) == 1, "the run never reached mflux's load()"
    err = capsys.readouterr().err
    assert f"{command}: error: the run failed (traceback above)." in err
    assert notes(err, command) == []


@SIDECAR_LEFT_OPEN_BY_MFLUX
@pytest.mark.parametrize(("adapter", "command"), COMMANDS)
@pytest.mark.parametrize(
    ("side", "expected"),
    [pytest.param(1024, "1024x1024", id="replay-1024"), pytest.param(512, None, id="replay-512")],
)
def test_a_metadata_replay_uses_the_restored_size(
    run_command, capsys, tmp_path: Path, adapter, command, side, expected
) -> None:
    """Bug: the notice reads the size before -C restores it (or ignores the restored size), so a replay of a
    1024x1024 run gets no note, or a replay of a 512x512 run gets one at the 1024x1024 default."""
    sidecar = tmp_path / "earlier.metadata.json"
    sidecar.write_text(json.dumps({"prompt": "a puffin", "seed": 3, "steps": 4, "width": side, "height": side}))
    run_command(["-C", str(sidecar), "--output", str(tmp_path / "replay.png")], adapter)
    assert notes(capsys.readouterr().err, command) == ([] if expected is None else [notice(command, expected)])
    assert FakeZImage.instances[0].generate_calls[0]["width"] == side


def test_base_with_an_init_image_prints_the_notice_at_its_default_size(run_command, capsys, small_png: Path) -> None:
    """Bug: --image silences the notice on the base command too, although its size never comes from the image: it
    runs at mflux's 1024x1024 default."""
    run_command([*BASE, "--seed", "1", "--image", str(small_png), "0.5"], z_image.ADAPTER)
    command = "mflux-generate-z-image-taef"
    assert notes(capsys.readouterr().err, command) == [notice(command, "1024x1024")]
    assert FakeZImage.instances[0].generate_calls[0]["width"] == 1024
