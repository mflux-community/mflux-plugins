"""Everything the plugin refuses happens before any weights load or any decoder is built, as exit 2 with a usable
message. Every refusal runs in _core, so each test runs both commands; the plain command each message names is
spelled out per command."""

import dataclasses
from types import SimpleNamespace

import pytest
from conftest import FAKES

from mflux.extras.taef import z_image, z_image_turbo

BASE = ["--prompt", "x", "--steps", "4"]
# (adapter, the plain mflux command its messages name)
COMMANDS = [
    pytest.param(z_image_turbo.ADAPTER, "mflux-generate-z-image-turbo", id="turbo"),
    pytest.param(z_image.ADAPTER, "mflux-generate-z-image", id="base"),
]


def refused(run_command, argv: list[str], capsys, adapter, *, expected: str) -> str:
    """Runs a command that must be refused before load: exit 2, no fake of any command built, and `expected` in the
    printed error (so an unrelated exit 2, such as an argparse error, does not pass). Returns stderr."""
    with pytest.raises(SystemExit) as exc:
        run_command(argv, adapter)
    assert exc.value.code == 2
    built = {fake.__name__: len(fake.instances) for _, fake in FAKES.values() if fake.instances}
    assert built == {}, "weights were loaded before the refusal"
    err = capsys.readouterr().err
    assert expected in err
    return err


@pytest.mark.parametrize(("adapter", "plain"), COMMANDS)
@pytest.mark.parametrize("value", ["0", "-3"])
def test_every_below_one_is_refused(run_command, capsys, value: str, adapter, plain) -> None:
    """Bug: --taef-every 0 or a negative cadence reaches the library (which would never write a preview, or divide by
    zero mid-run), the check runs after load, or the hint names the other command."""
    refused(
        run_command,
        [*BASE, "--taef-every", value],
        capsys,
        adapter,
        expected=f"--taef-every must be 1 or more, got {value}. Run {plain} to generate without previews.",
    )


@pytest.mark.parametrize(("adapter", "plain"), COMMANDS)
def test_every_with_frames_is_refused(run_command, capsys, adapter, plain) -> None:
    """Bug: --taef-every 3 --taef-frames runs, and the library silently forces every step, so the user gets three
    times the frames they asked for."""
    refused(
        run_command,
        [*BASE, "--taef-every", "3", "--taef-frames"],
        capsys,
        adapter,
        expected="--taef-every and --taef-frames can't be used together: --taef-frames writes every step.",
    )


@pytest.mark.parametrize(("adapter", "plain"), COMMANDS)
@pytest.mark.parametrize("path", ["p.gif", "p", "out/"])
def test_unknown_suffix_is_refused(run_command, capsys, path: str, adapter, plain) -> None:
    """Bug: a preview path without an image suffix (or with one Pillow writes but no viewer refreshes, like .gif) is
    accepted, and the first preview write fails after the model has loaded."""
    refused(
        run_command,
        [*BASE, "--taef-preview", path],
        capsys,
        adapter,
        expected=(
            f"--taef-preview must end in .png, .jpg, .jpeg or .webp, got {path}. "
            f"Run {plain} to generate without previews."
        ),
    )


@pytest.mark.parametrize(("adapter", "plain"), COMMANDS)
def test_an_empty_frames_path_for_several_seeds_is_refused_not_a_traceback(run_command, capsys, adapter, plain) -> None:
    """Bug: the suffix check builds the per-seed frames path first, and an empty --taef-preview has no stem to add
    _seed_ to, so the run ends in a ValueError traceback instead of the refusal."""
    refused(
        run_command,
        [*BASE, "--seed", "1", "2", "--taef-frames", "--taef-preview", ""],
        capsys,
        adapter,
        expected="--taef-preview must end in .png, .jpg, .jpeg or .webp",
    )


@pytest.mark.parametrize(("adapter", "plain"), COMMANDS)
@pytest.mark.parametrize("path", ["{step}.png", "{}.png", "{seed.png", "a}.png", "{seed.real}.png"])
def test_format_fields_other_than_seed_are_refused(run_command, capsys, path: str, adapter, plain) -> None:
    """Bug: a brace field other than {seed} (or an unclosed brace) passes the checks and the run ends in a KeyError
    or ValueError traceback when the path is formatted, possibly after the model has loaded."""
    refused(
        run_command,
        [*BASE, "--taef-preview", path],
        capsys,
        adapter,
        expected=(
            f"--taef-preview can contain {{seed}} and no other {{...}} field, got {path}. "
            f"Run {plain} to generate without previews."
        ),
    )


@pytest.mark.parametrize(("adapter", "plain"), COMMANDS)
@pytest.mark.parametrize(
    "path",
    [
        "{seed:x.y}.png",
        "{seed:s}.png",
        "{seed!z}.png",
        "{seed:{w}}.png",
        "{seed:{}}.png",
        "{seed:{seed.foo}}.png",
        "{seed:{seed[0]}}.png",
    ],
)
def test_a_seed_field_python_cannot_format_is_refused(run_command, capsys, path: str, adapter, plain) -> None:
    """Bug: the field check reads only the field's name, so {seed} with a format spec or conversion a seed can't take
    (or a field nested in its spec) passes it, and the run ends in a ValueError, KeyError, IndexError,
    AttributeError or TypeError traceback when the path is formatted."""
    refused(
        run_command,
        [*BASE, "--taef-preview", path],
        capsys,
        adapter,
        expected=f"--taef-preview can contain {{seed}} and no other {{...}} field, got {path}.",
    )


@pytest.mark.parametrize(("adapter", "plain"), COMMANDS)
@pytest.mark.parametrize(
    ("preview", "output", "seeds"),
    [
        pytest.param("a.png", "a.png", ["1"], id="equal"),
        pytest.param("./a.png", "a.png", ["1"], id="dot-slash"),
        # A default APFS volume is case-insensitive, so this is one file. Neither file exists when the command
        # parses, so the comparison must fold case itself on any volume.
        pytest.param("A.PNG", "a.png", ["1"], id="case"),
        # The one live file is seed 2's output (mflux names it a_seed_2.png).
        pytest.param("a_seed_2.png", "a.png", ["1", "2"], id="second-seed"),
    ],
)
def test_preview_equal_to_output_is_refused(run_command, capsys, preview, output, seeds, adapter, plain) -> None:
    """Bug: the refusal compares exact strings (or only the first seed), so the last preview holds a.png and mflux,
    whose save never overwrites, writes the final image as a_1.png."""
    refused(
        run_command,
        [*BASE, "--taef-preview", preview, "--output", output, "--seed", *seeds],
        capsys,
        adapter,
        expected="--taef-preview names the same file as --output",
    )


@pytest.mark.parametrize(("adapter", "plain"), COMMANDS)
@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["--taef-every", "1"], id="every-1"),
        pytest.param(["--taef-every", "3"], id="every-3"),
        pytest.param(["--taef-frames"], id="frames"),
        pytest.param(["--taef-preview", "p/{seed}.webp"], id="seed-field"),
        pytest.param(["--taef-preview", "LIVE.JPEG"], id="upper-case-suffix"),
        pytest.param(["--taef-preview", "live.jpg", "--output", "live.png"], id="same-stem-other-suffix"),
        pytest.param(["--taef-preview", "{{seed}}.png"], id="escaped-braces"),
        pytest.param(["--taef-preview", "p_{seed:04d}.png"], id="seed-format-spec"),
    ],
)
def test_valid_preview_options_reach_the_model(run_command, argv, adapter, plain) -> None:
    """Bug: a refusal is too wide: --taef-every 1 (the lowest valid cadence) is refused by an off-by-one check, an
    upper-case suffix by a case-sensitive one, a {seed} field, escaped braces or a format spec a seed takes by the
    field check, or a preview that shares only the output's stem by the same-file check."""
    run_command([*BASE, "--seed", "1", *argv], adapter)
    (fake,) = {fake for _, fake in FAKES.values()}
    assert len(fake.instances) == 1


@pytest.mark.parametrize(("adapter", "plain"), COMMANDS)
def test_unsupported_model_config_is_refused(run_command, capsys, adapter, plain) -> None:
    """Bug: the command does not ask mlx-taef which decoder fits before load, so a model mlx-taef has no decoder for
    loads its weights and fails afterwards (or the UnsupportedMfluxModelError escapes as a traceback), or the hint
    names the other command."""
    adapter = dataclasses.replace(
        adapter, validate=lambda args: SimpleNamespace(model_name="acme/other", base_model=None, aliases=[])
    )
    err = refused(run_command, [*BASE, "--seed", "1"], capsys, adapter, expected="TAEF has no preview decoder for")
    assert f"TAEF has no preview decoder for acme/other. Run {plain} to generate without previews." in err


@pytest.mark.parametrize(
    ("adapter", "model", "expected"),
    [
        pytest.param(
            z_image_turbo.ADAPTER,
            "z-image",
            "'z-image' is not Tongyi-MAI/Z-Image-Turbo; this CLI only accepts the aliases",
            id="turbo",
        ),
        # The ControlNet entry has its own mflux command; the base command runs only z-image and z-image-turbo.
        pytest.param(
            z_image.ADAPTER,
            "z-image-controlnet",
            "'z-image-controlnet' is not Tongyi-MAI/Z-Image; this CLI only accepts the aliases",
            id="base",
        ),
    ],
)
def test_a_family_outside_the_command_is_refused_by_mflux(run_command, capsys, adapter, model, expected) -> None:
    """Bug: run() stops turning validate's ModelConfigError into a usage error, so a model outside the command's
    family (--model z-image on the Turbo command, the ControlNet entry on the base command) ends in a traceback (or
    loads the wrong model) instead of exit 2."""
    refused(run_command, [*BASE, "--seed", "1", "--model", model], capsys, adapter, expected=expected)
