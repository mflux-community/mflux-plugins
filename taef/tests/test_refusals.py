"""Everything the plugin refuses happens before any weights load or any decoder is built, as exit 2 with a usable
message. Every refusal runs in _core, so each test runs both commands; the plain command each message names is
spelled out per command."""

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest
from _fakes import FakeLivePreview
from conftest import FAKES

from mflux.extras.taef import z_image, z_image_turbo

BASE = ["--prompt", "x", "--steps", "4"]
# (adapter, the plain mflux command its messages name)
COMMANDS = [
    pytest.param(z_image_turbo.ADAPTER, "mflux-generate-z-image-turbo", id="turbo"),
    pytest.param(z_image.ADAPTER, "mflux-generate-z-image", id="base"),
]


def refused(run_command, argv: list[str], capsys, adapter, *, expected: str) -> str:
    """Runs a command that must be refused before load: exit 2, no fake of any command built, no preview decoder
    built, and `expected` in the printed error (so an unrelated exit 2, such as an argparse error, does not pass).
    Returns stderr."""
    with pytest.raises(SystemExit) as exc:
        run_command(argv, adapter)
    assert exc.value.code == 2
    built = {fake.__name__: len(fake.instances) for _, fake in FAKES.values() if fake.instances}
    assert built == {}, "weights were loaded before the refusal"
    assert FakeLivePreview.instances == [], "the preview decoder was built before the refusal"
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
    "path",
    [
        "{seed!r}.png",
        "{seed!s}.png",
        # Seed 47 formats to "/", which puts the preview in another folder.
        "x{seed:c}.png",
        "{seed:x}.png",
        "{seed:>8}.png",
        "{seed:+}.png",
        "{seed:,}.png",
        "{seed:d}.png",
        "{seed:04dd}.png",
        # A width with a type letter other than d.
        "x{seed:3c}.png",
        "{seed:4x}.png",
        "{seed:08b}.png",
        # Widths of four digits: one more than the largest allowed width, with and without the zero and the d.
        "{seed:1000}.png",
        "{seed:01000}.png",
        "{seed:1000d}.png",
        # A leading zero counts as one of the three digits.
        "{seed:0004}.png",
        "{seed:0999}.png",
        "{seed:0100d}.png",
    ],
)
def test_a_seed_spec_other_than_a_zero_padded_width_is_refused(run_command, capsys, path: str, adapter, plain) -> None:
    """Bug: any format spec Python's int accepts passes, so {seed:>2000000000} builds a 2 GB path before load and
    {seed:c} turns seed 47 into a folder separator; or a conversion such as {seed!r} slips through."""
    refused(
        run_command,
        [*BASE, "--seed", "47", "--taef-preview", path],
        capsys,
        adapter,
        expected=(
            f"--taef-preview can contain {{seed}} and no other {{...}} field, got {path}. "
            f"Run {plain} to generate without previews."
        ),
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
    ("preview", "output"),
    [
        pytest.param("p.png", "p_step00.png", id="first-frame"),
        pytest.param("p.png", "p_step123.png", id="three-digit-step"),
        # A default APFS volume is case-insensitive, so this is one file.
        pytest.param("p.png", "P_STEP03.PNG", id="case"),
        pytest.param("sub/p.png", "sub/./p_step01.png", id="same-folder-other-spelling"),
        # The folders differ only in case: one folder on a default APFS volume.
        pytest.param("sub/p.png", "Sub/p_step01.png", id="same-folder-other-case"),
    ],
)
def test_a_frame_named_like_the_output_is_refused(run_command, capsys, preview, output, adapter, plain) -> None:
    """Bug: in frames mode the same-file check compares the output with the preview's base name, which frames mode
    never writes, so --output p_step00.png --taef-preview p.png --taef-frames passes; the first frame holds the name
    and mflux, whose save never overwrites, writes the final image as p_step00_1.png."""
    refused(
        run_command,
        [*BASE, "--seed", "1", "--taef-frames", "--taef-preview", preview, "--output", output],
        capsys,
        adapter,
        expected="names the same file as --output",
    )


@pytest.mark.parametrize(("adapter", "plain"), COMMANDS)
@pytest.mark.parametrize(
    ("preview", "output"),
    [
        # Frames mode writes a_stepNN.png and never a.png itself.
        pytest.param("a.png", "a.png", id="base-name"),
        pytest.param("p.png", "p_step.png", id="no-step-digits"),
        pytest.param("p.png", "p_step00.jpg", id="other-suffix"),
        pytest.param("p.png", "other/p_step00.png", id="other-folder"),
        pytest.param("p.png", "q_step00.png", id="other-stem"),
        pytest.param("p.png", "p_step00x.png", id="text-after-the-digits"),
        # The output needs a suffix mflux's save takes, so the text after the frame suffix ends in one.
        pytest.param("p.png", "p_step00.png.jpg", id="text-after-the-suffix"),
    ],
)
def test_frames_that_cannot_name_the_output_reach_the_model(run_command, preview, output, adapter, plain) -> None:
    """Bug: the frames-mode check is too wide: it still refuses the preview's base name, which frames mode never
    writes, or it matches a frame name without its digits, suffix or folder."""
    run_command([*BASE, "--seed", "1", "--taef-frames", "--taef-preview", preview, "--output", output], adapter)
    (fake,) = {fake for _, fake in FAKES.values()}
    assert len(fake.instances) == 1


@pytest.mark.parametrize(("adapter", "plain"), COMMANDS)
@pytest.mark.parametrize(
    ("make", "argv"),
    [
        pytest.param(lambda: Path("p.png").mkdir(), ["--taef-preview", "p.png"], id="directory"),
        pytest.param(lambda: Path("p.png").mkdir(), ["--taef-preview", "p.png", "--taef-frames"], id="directory-frames"),
        pytest.param(lambda: Path("image.preview.png").mkdir(), [], id="default-path-directory"),
        pytest.param(lambda: Path("p").mkdir() or Path("p/1.webp").mkdir(), ["--taef-preview", "p/{seed}.webp"], id="seed"),
        pytest.param(lambda: Path("p.png").symlink_to("elsewhere.png"), ["--taef-preview", "p.png"], id="dangling-link"),
        pytest.param(
            lambda: Path("elsewhere.png").touch() or Path("p.png").symlink_to("elsewhere.png"),
            ["--taef-preview", "p.png"],
            id="link-to-a-file",
        ),
    ],
)  # fmt: skip
def test_a_preview_path_that_is_a_directory_or_a_link_is_refused(
    run_command, capsys, make, argv, adapter, plain
) -> None:
    """Bug: a preview path that is an existing directory passes the suffix check and the first preview write fails
    only after the model has loaded; or a symbolic link is followed and the preview overwrites the file it points to."""
    make()
    refused(
        run_command,
        [*BASE, "--seed", "1", *argv],
        capsys,
        adapter,
        expected="--taef-preview must name a file, not a directory or a symbolic link",
    )


@pytest.mark.parametrize(("adapter", "plain"), COMMANDS)
def test_a_frame_named_like_an_absolute_output_is_refused(run_command, capsys, adapter, plain) -> None:
    """Bug: the frames-mode check compares the folders as typed, so a relative --taef-preview p.png and an absolute
    --output naming p_step01.png in the same folder pass; the frame holds the name and the final image goes to
    p_step01_1.png."""
    output = str(Path.cwd() / "p_step01.png")
    refused(
        run_command,
        [*BASE, "--seed", "1", "--taef-frames", "--taef-preview", "p.png", "--output", output],
        capsys,
        adapter,
        expected="names the same file as --output",
    )


@pytest.mark.parametrize(("adapter", "plain"), COMMANDS)
@pytest.mark.parametrize("seeds", [["2", "12"], ["12", "2"]], ids=["preview-seed-first", "output-seed-first"])
def test_a_preview_named_like_another_seeds_output_is_refused(run_command, capsys, seeds, adapter, plain) -> None:
    """Bug: the same-file check compares each seed's preview only with that seed's output, so seed 2's preview
    a_seed_12.png is seed 12's output; whichever is written second, mflux's save, which never overwrites, writes seed
    12's image as a_seed_12_1.png or the preview replaces seed 12's image."""
    refused(
        run_command,
        [*BASE, "--seed", *seeds, "--output", "a.png", "--taef-preview", "a_seed_1{seed}.png"],
        capsys,
        adapter,
        expected="--taef-preview names the same file as --output (a_seed_12.png). Give the preview its own name.",
    )


def _steps(n: int) -> list[str]:
    return ["--prompt", "x", "--steps", str(n)]


# 252 + 4 bytes; each é is two bytes in UTF-8, so the second name is 256 bytes in 130 characters.
LONG_NAME_REFUSALS = [
    pytest.param([*_steps(4), "--taef-preview", "a" * 252 + ".png"], "a" * 252 + ".png", 256, id="256-bytes"),
    pytest.param([*_steps(4), "--taef-preview", "é" * 126 + ".png"], "é" * 126 + ".png", 256, id="counted-in-bytes"),
    pytest.param([*_steps(4), "--taef-preview", "p_{seed:999}.png"], None, 1005, id="seed-width-999"),
    # The default preview adds .preview to the output's stem: 244 + 12 bytes.
    pytest.param([*_steps(4), "--output", "a" * 244 + ".png"], "a" * 244 + ".preview.png", 256, id="default-path"),
    # 100 steps write frames up to _step99; 101 steps reach _step100, 244 + 8 + 4 bytes.
    pytest.param(
        [*_steps(101), "--taef-frames", "--taef-preview", "a" * 244 + ".png"],
        "a" * 244 + "_step100.png",
        256,
        id="longest-frame-name",
    ),
]


@pytest.mark.parametrize(("adapter", "plain"), COMMANDS)
@pytest.mark.parametrize(("argv", "name", "size"), LONG_NAME_REFUSALS)
def test_a_preview_name_over_255_bytes_is_refused(run_command, capsys, argv, name, size, adapter, plain) -> None:
    """Bug: a preview file name over the 255 bytes a file name may have passes the checks (a {seed:999} width, a long
    --output, or in frames mode the longest frame name), the model loads, and the first save fails with ENAMETOOLONG,
    which turns the preview off without a word; or the check counts characters, not bytes."""
    err = refused(
        run_command,
        [*argv, "--seed", "1"],
        capsys,
        adapter,
        expected=f"--taef-preview makes a file name of {size} bytes, over the 255 a file name may have: ",
    )
    if name is not None:
        assert f"{name}. Give the preview a shorter name." in err


@pytest.mark.parametrize(("adapter", "plain"), COMMANDS)
@pytest.mark.parametrize(
    "argv",
    [
        pytest.param([*_steps(4), "--taef-preview", "a" * 251 + ".png"], id="255-bytes"),
        pytest.param([*_steps(4), "--output", "a" * 243 + ".png"], id="default-path-255-bytes"),
        pytest.param([*_steps(100), "--taef-frames", "--taef-preview", "a" * 244 + ".png"], id="longest-frame-255-bytes"),
    ],
)  # fmt: skip
def test_a_preview_name_of_255_bytes_reaches_the_model(run_command, argv, adapter, plain) -> None:
    """Bug: the length check is off by one (it refuses a 255-byte name), or counts a frame name for one step more
    than the run writes (_step100 for a 100-step run, whose last frame is _step99)."""
    run_command([*argv, "--seed", "1"], adapter)
    (fake,) = {fake for _, fake in FAKES.values()}
    assert len(fake.instances) == 1


@pytest.mark.parametrize(("adapter", "plain"), COMMANDS)
@pytest.mark.parametrize(
    ("make", "argv", "shown"),
    [
        pytest.param(lambda: Path("p_step00.png").mkdir(), ["--taef-preview", "p.png"], "p_step00.png", id="directory-at-the-first-frame"),
        pytest.param(lambda: Path("p_step03.png").symlink_to("elsewhere.png"), ["--taef-preview", "p.png"], "p_step03.png", id="dangling-link-at-a-frame"),
        pytest.param(
            lambda: Path("elsewhere.png").touch() or Path("p_step12.png").symlink_to("elsewhere.png"),
            ["--taef-preview", "p.png"],
            "p_step12.png",
            id="link-to-a-file-at-a-frame",
        ),
        # A default APFS volume is case-insensitive, so this is the folder the first frame would be written into.
        pytest.param(lambda: Path("sub").mkdir() or Path("sub/P_STEP00.PNG").mkdir(), ["--taef-preview", "sub/p.png"], "sub/P_STEP00.PNG", id="other-case"),
        pytest.param(lambda: Path("image.preview_step00.png").mkdir(), [], "image.preview_step00.png", id="default-path"),
        # Every seed's frames are checked, not only the first seed's.
        pytest.param(lambda: Path("p_seed_2_step00.png").mkdir(), ["--seed", "1", "2", "--taef-preview", "p.png"], "p_seed_2_step00.png", id="second-seed"),
    ],
)  # fmt: skip
def test_a_frame_name_that_is_a_directory_or_a_link_is_refused(
    run_command, capsys, make, argv, shown, adapter, plain
) -> None:
    """Bug: frames mode checks only the preview's base name, which it never writes, so a symbolic link at a frame name
    is followed and the frame overwrites the file it points to, or a directory there makes the first frame fail
    after the model has loaded."""
    make()
    refused(
        run_command,
        [*BASE, "--seed", "1", "--taef-frames", *argv],
        capsys,
        adapter,
        expected=f"--taef-preview frames must be files, not a directory or a symbolic link, got {shown}.",
    )


@pytest.mark.parametrize(("adapter", "plain"), COMMANDS)
@pytest.mark.parametrize(
    ("make", "argv"),
    [
        # A second run overwrites the first run's frames.
        pytest.param(lambda: Path("p_step00.png").touch(), ["--taef-frames"], id="frame-file"),
        # Without --taef-frames no frame is written.
        pytest.param(lambda: Path("p_step00.png").mkdir(), [], id="directory-without-frames"),
        pytest.param(lambda: Path("p_step00.png.bak").mkdir(), ["--taef-frames"], id="text-after-the-suffix"),
        pytest.param(lambda: Path("q_step00.png").mkdir(), ["--taef-frames"], id="other-stem"),
    ],
)
def test_frame_names_the_run_can_write_reach_the_model(run_command, make, argv, adapter, plain) -> None:
    """Bug: the frame-name check refuses an ordinary file left by an earlier run, runs outside frames mode, or matches
    a name no frame takes (text after the suffix, another stem)."""
    make()
    run_command([*BASE, "--seed", "1", "--taef-preview", "p.png", *argv], adapter)
    (fake,) = {fake for _, fake in FAKES.values()}
    assert len(fake.instances) == 1


@pytest.mark.parametrize(("adapter", "plain"), COMMANDS)
def test_an_existing_preview_file_is_not_refused(run_command, adapter, plain) -> None:
    """Bug: the directory and link check refuses any path that exists, so a second run with the same preview path
    is refused instead of overwriting the old preview."""
    Path("live.png").touch()
    run_command([*BASE, "--seed", "1", "--taef-preview", "live.png"], adapter)
    (fake,) = {fake for _, fake in FAKES.values()}
    assert len(fake.instances) == 1


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
        pytest.param(["--taef-preview", "p_{seed:04}.png"], id="seed-zero-padded-width"),
        pytest.param(["--taef-preview", "p_{seed:4}.png"], id="seed-width"),
        # The largest widths allowed: three digits in all, a leading zero counted, with and without the d.
        pytest.param(["--taef-preview", "p_{seed:100}.png"], id="seed-three-digit-width"),
        pytest.param(["--taef-preview", "p_{seed:099d}.png"], id="seed-zero-padded-three-digit-width"),
    ],
)
def test_valid_preview_options_reach_the_model(run_command, argv, adapter, plain) -> None:
    """Bug: a refusal is too wide: --taef-every 1 (the lowest valid cadence) is refused by an off-by-one check, an
    upper-case suffix by a case-sensitive one, a {seed} field, escaped braces or a width (zero-padded or not, up to
    three digits, with or without d) by the field check, or a preview that shares only the output's stem by the
    same-file check."""
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
