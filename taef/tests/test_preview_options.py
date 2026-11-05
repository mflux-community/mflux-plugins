"""Where each seed's preview goes, and what the three preview flags turn into, read through the real parser."""

import sys
from argparse import Namespace
from pathlib import Path

import pytest

from mflux.extras.taef import _core, z_image_turbo


def parsed(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> tuple[object, Namespace]:
    """The Turbo command's parser and what it makes of `argv` (mflux's parse_args adds _seed_{seed} to --output for
    several seeds)."""
    monkeypatch.setattr(sys, "argv", [z_image_turbo.ADAPTER.command, "--prompt", "x", *argv])
    parser = z_image_turbo.build_parser()
    return parser, parser.parse_args()


def preview_paths(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> list[Path]:
    _, args = parsed(monkeypatch, argv)
    return [_core.preview_path(args, seed) for seed in args.seed]


def test_default_preview_sits_next_to_the_output(monkeypatch) -> None:
    """Bug: the default preview goes to the working directory, or replaces the output's suffix instead of adding
    .preview to its name."""
    assert preview_paths(monkeypatch, ["--output", "out/img.png", "--seed", "42"]) == [Path("out/img.preview.png")]


@pytest.mark.parametrize("frames", [[], ["--taef-frames"]], ids=["live", "frames"])
def test_default_preview_follows_mflux_seed_names(monkeypatch, frames) -> None:
    """Bug: frames mode adds _seed_ again to a default path that already carries mflux's suffix
    (image_seed_1_seed_1.preview.png), or the default path ignores the seed and two seeds share one file."""
    assert preview_paths(monkeypatch, ["--seed", "1", "2", *frames]) == [
        Path("image_seed_1.preview.png"),
        Path("image_seed_2.preview.png"),
    ]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        pytest.param(["--seed", "7"], ["p/7.webp"], id="one-seed"),
        pytest.param(["--seed", "1", "2"], ["p/1.webp", "p/2.webp"], id="two-seeds"),
        pytest.param(["--seed", "1", "2", "--taef-frames"], ["p/1.webp", "p/2.webp"], id="two-seeds-frames"),
    ],
)
def test_explicit_path_formats_seed(monkeypatch, argv, expected) -> None:
    """Bug: {seed} in --taef-preview stops being filled in, or _seed_ is added next to it in frames mode
    (p/1_seed_1.webp)."""
    assert preview_paths(monkeypatch, ["--taef-preview", "p/{seed}.webp", *argv]) == [Path(p) for p in expected]


@pytest.mark.parametrize(
    ("seeds", "expected"),
    [pytest.param(["1", "2"], ["live.png", "live.png"], id="two-seeds"), pytest.param(["1"], ["live.png"], id="one")],
)
def test_explicit_path_without_seed_stays_one_file_for_several_seeds(monkeypatch, seeds, expected) -> None:
    """Bug: the seed suffix is added outside frames mode, so a viewer left open on live.png stops updating after the
    first seed."""
    assert preview_paths(monkeypatch, ["--taef-preview", "live.png", "--seed", *seeds]) == [Path(p) for p in expected]


@pytest.mark.parametrize(
    ("seeds", "expected"),
    [
        pytest.param(["1", "2"], ["live_seed_1.png", "live_seed_2.png"], id="two-seeds"),
        pytest.param(["1"], ["live.png"], id="one"),
    ],
)
def test_explicit_frames_path_without_seed_gets_seed_suffix_for_several_seeds(monkeypatch, seeds, expected) -> None:
    """Bug: the suffix is dropped in frames mode, so seed 2's frames overwrite seed 1's; or it is added for a single
    seed, where nothing can collide."""
    argv = ["--taef-frames", "--taef-preview", "live.png", "--seed", *seeds]
    assert preview_paths(monkeypatch, argv) == [Path(p) for p in expected]


def test_unset_flags_mean_every_step_one_file(monkeypatch) -> None:
    """Bug: an unset --taef-every reaches the library as None (or 0), or frames default to on."""
    parser, args = parsed(monkeypatch, ["--seed", "1"])
    options = _core.checked_preview_options(parser, z_image_turbo.ADAPTER, args)
    assert options == _core.PreviewOptions(every=1, frames=False)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        pytest.param(["--taef-every", "3"], _core.PreviewOptions(every=3, frames=False), id="every"),
        pytest.param(["--taef-frames"], _core.PreviewOptions(every=1, frames=True), id="frames"),
    ],
)
def test_set_flags_reach_the_options(monkeypatch, argv, expected) -> None:
    """Bug: the options keep their defaults whatever the command line says (--taef-every 3 decodes every step, or
    --taef-frames keeps overwriting one file)."""
    parser, args = parsed(monkeypatch, ["--seed", "1", *argv])
    assert _core.checked_preview_options(parser, z_image_turbo.ADAPTER, args) == expected


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        pytest.param("a.png", "a.png", True, id="equal"),
        pytest.param("./a.png", "a.png", True, id="dot-slash"),
        pytest.param("A.PNG", "a.png", True, id="case"),
        pytest.param("d/../a.png", "a.png", True, id="dot-dot"),
        pytest.param("a.png", "b.png", False, id="other-name"),
        pytest.param("x/a.png", "a.png", False, id="other-folder"),
    ],
)
def test_same_file_key_matches_one_file_on_a_case_insensitive_volume(a, b, same) -> None:
    """Bug: the key compares the strings as typed (./a.png and a.png differ) or keeps their case (A.PNG and a.png
    differ, though a default APFS volume stores them as one file); or it folds away the folder, so x/a.png and a.png
    count as one file."""
    assert (_core.same_file_key(Path(a)) == _core.same_file_key(Path(b))) is same
