"""Pins the parts of mflux's Z-Image command this package calls or copies.

Every pinned value is recorded once per mflux source it was reviewed against, under a label that names that source
(a release, or a main commit and the review date). A value no label holds fails. When one fails after an mflux
update: read mflux's diff of src/mflux/models/z_image/cli/z_image_generate.py, bring z_image.py (and _core.run, if
the flow after load() changed) in line with it, then add the values the failure messages print under a new label."""

import builtins
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import sys
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from _fakes import FakeZImage
from _fingerprint import ast_fingerprint, function_body_dumps, statements_fingerprint_from
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.z_image.cli import z_image_generate
from mflux.models.z_image.cli.z_image_generate import ZImageCommand
from mflux.models.z_image.variants.z_image import ZImage
from mflux.utils.apple_silicon import AppleSiliconUtil
from mflux.utils.generated_image import GeneratedImage

import mflux
from mflux.extras.myplugin import z_image

# mflux main a3b21fc added VAEDecodeOptions (the --vae-full-precision decode, and a warning that --vae-tiling no
# longer changes the decode); the mflux 0.22.0 / 0.23.0 releases have neither.
HAS_DECODE_OPTIONS = importlib.util.find_spec("mflux.models.common.vae.vae_decode_options") is not None

# The reviewed mflux sources. The released digests were recorded on PyPI 0.22.0 (commit 31c9640); v.0.23.0 has the
# same z_image_generate.py, variants/z_image.py, cli/parser/parsers.py and utils/apple_silicon.py. CI checks every
# digest on CPython 3.11 and 3.13.
RELEASED = "mflux v.0.22.0 / v.0.23.0 released"
# main a3b21fc: --vae-full-precision in build_parser, VAEDecodeOptions.warn_encode_only_tiling() in main() (mirrored
# by z_image.warn_before_load), VAEDecodeOptions.generation_parameters in ZImage.generate_image (mirrored by
# FakeZImage). ZImageCommand.load applies the decode mode itself, and the flow after load() is unchanged.
MAIN = "mflux main a3b21fc, reviewed 2026-10-23"
# PyPI 0.24.0 (tag v.0.24.0, commit 8a7fd0a), reviewed 2026-10-28 on CPython 3.13. Against main a3b21fc it leaves
# z_image_generate.py, variants/z_image.py, utils/apple_silicon.py and VAEDecodeOptions unchanged. Its parsers.py adds
# add_cfg_end_arguments (not called by the Z-Image parser), a bake_by_default parameter on add_lora_arguments (default
# True, as before) and a -C replay of bake_lora; none of it moves a pin here.
RELEASED_0_24 = "mflux v.0.24.0 released"
# The versions each reviewed source reports. mflux main a3b21fc still reports 0.23.0, so the version alone cannot tell
# it from the release; a release that ships main's code reports a version its label does not list.
REPORTED_VERSIONS = {RELEASED: ("0.22.0", "0.23.0"), MAIN: ("0.23.0",), RELEASED_0_24: ("0.24.0",)}

BUILD_PARSER_DIGESTS = {RELEASED: "664a51c06e0e4b7a", MAIN: "6a85a43cc52fd44f", RELEASED_0_24: "6a85a43cc52fd44f"}
MAIN_DIGESTS = {RELEASED: "38cfe3fe3f1142f7", MAIN: "1f3058eda927d0fe", RELEASED_0_24: "1f3058eda927d0fe"}
# ZImageCommand.load, which _core.run calls for the model. main a3b21fc: it binds the ZImage it builds, calls
# VAEDecodeOptions.apply(args, model) and returns the model (the released load returns ZImage(...) directly).
LOAD_DIGESTS = {RELEASED: "658f557e2fa63a64", MAIN: "d91dfaa735845a7b", RELEASED_0_24: "d91dfaa735845a7b"}
# ZImage.generate_image, the method tests/_fakes.py's FakeZImage.generate_image reproduces up to the image record.
GENERATE_IMAGE_DIGESTS = {RELEASED: "a0e64be2aa3d8bc9", MAIN: "88204535dc85b78e", RELEASED_0_24: "88204535dc85b78e"}
# ZImage._predict, the factory a plugin replaces to keep per-step state, and the chip check that decides whether mflux
# compiles the step it returns. AGENTS.md's "Compile and memory" section describes both.
PREDICT_DIGESTS = {
    RELEASED: ("016c64c92ceefbdf", "3264a79fbfa0f1a5"),
    MAIN: ("016c64c92ceefbdf", "3264a79fbfa0f1a5"),
    RELEASED_0_24: ("016c64c92ceefbdf", "3264a79fbfa0f1a5"),
}

# Every option string and dest of the Z-Image parser, sorted. build_parser's digest sees only the helper calls it
# makes; an option added inside a shared helper (add_image_generator_arguments and the like) shows up only here.
RELEASED_OPTION_STRINGS = (
    "--auto-seeds", "--bake-lora", "--base-model", "--battery-percentage-stop-limit", "--compute-precision",
    "--config-from-conf", "--config-from-metadata", "--float32", "--guidance", "--height", "--help", "--image",
    "--image-path", "--image-strength", "--lora", "--lora-paths", "--lora-scales", "--lora-style", "--low-ram",
    "--make-conf", "--metadata", "--mlx-cache-limit-gb", "--model", "--negative-prompt", "--no-bake-lora",
    "--no-exif", "--no-float32", "--no-metadata", "--no-pid-decode", "--output", "--pid-decode",
    "--pid-degrade-sigma", "--prompt", "--prompt-file", "--quantize", "--scheduler", "--seed", "--steps",
    "--stepwise-image-output-dir", "--vae-tile-size", "--vae-tiling", "--verbose", "--width", "-B", "-C", "-h", "-m",
    "-q", "-v",
)  # fmt: skip
RELEASED_DESTS = (
    "auto_seeds", "bake_lora", "base_model", "battery_percentage_stop_limit", "compute_precision",
    "config_from_metadata", "float32", "guidance", "height", "help", "image", "image_path", "image_strength", "lora",
    "lora_paths", "lora_scales", "lora_style", "low_ram", "metadata", "mlx_cache_limit_gb", "model", "negative_prompt",
    "no_metadata", "output", "pid_decode", "pid_degrade_sigma", "prompt", "prompt_file", "quantize", "scheduler",
    "seed", "steps", "stepwise_image_output_dir", "vae_tile_size", "vae_tiling", "verbose", "width",
)  # fmt: skip
# main a3b21fc: add_vae_full_precision_arguments() adds one BooleanOptionalAction. 0.24.0 has the same options.
VAE_FULL_PRECISION_OPTIONS = (
    tuple(sorted((*RELEASED_OPTION_STRINGS, "--vae-full-precision", "--no-vae-full-precision"))),
    tuple(sorted((*RELEASED_DESTS, "vae_full_precision"))),
)
OPTIONS = {
    RELEASED: (RELEASED_OPTION_STRINGS, RELEASED_DESTS),
    MAIN: VAE_FULL_PRECISION_OPTIONS,
    RELEASED_0_24: VAE_FULL_PRECISION_OPTIONS,
}

# Per adapter: the mirrored mflux main(), its first statement after the pre-load block, and the digest of everything
# from there on (load, callback registration, the per-seed generate/save loop, the except/finally). _core.run mirrors
# that tail; a new adapter adds its row here.
POST_LOAD = {
    "mflux-generate-z-image-myplugin": (
        z_image_generate.main,
        "model = ZImageCommand.load(args)",
        {RELEASED: "812cb2e320c4eaee", MAIN: "812cb2e320c4eaee", RELEASED_0_24: "812cb2e320c4eaee"},
    ),
}

REVIEW_FIRST = (
    "review the mflux change first (AGENTS.md, 'When the drift test fails'), then record this value under a label "
    "naming the mflux source you reviewed:"
)


def reviewed_sources(accepted: Mapping[str, object], actual: object) -> set[str]:
    """The labels of the reviewed mflux sources whose recorded value is `actual`; empty for an unreviewed one."""
    return {source for source, value in accepted.items() if value == actual}


def parser_options() -> tuple[tuple[str, ...], tuple[str, ...]]:
    parser = z_image_generate.build_parser()
    strings = tuple(sorted(option for action in parser._actions for option in action.option_strings))
    dests = tuple(sorted({action.dest for action in parser._actions}))
    return strings, dests


def prediction_step_digests() -> tuple[str, str]:
    return ast_fingerprint(ZImage._predict), ast_fingerprint(AppleSiliconUtil.is_m1_or_m2)


def test_mflux_build_parser_is_a_reviewed_one() -> None:
    """Bug: mflux's Z-Image parser gains or drops an option (a step-cache flag, say) and the plugin
    keeps passing through a parser nobody checked against its refusals."""
    actual = ast_fingerprint(z_image_generate.build_parser)
    assert reviewed_sources(BUILD_PARSER_DIGESTS, actual), (
        f"z_image_generate.build_parser changed: {REVIEW_FIRST} {actual}"
    )


def test_mflux_parser_options_are_reviewed_ones() -> None:
    """Bug: mflux adds an option through a shared parser helper (a step-cache flag on Z-Image, say);
    build_parser's own digest does not move, and the plugin passes the flag through without a refusal."""
    strings, dests = parser_options()
    assert reviewed_sources(OPTIONS, (strings, dests)), (
        f"z_image_generate.build_parser() options changed: {REVIEW_FIRST}\n{strings}\n{dests}"
    )


@pytest.mark.parametrize("command", sorted(POST_LOAD))
def test_the_post_load_flow_each_adapter_mirrors_is_a_reviewed_one(command: str) -> None:
    """Bug: mflux changes what its command does after load() (another callback, a different save call, a new except
    clause) and _core.run keeps mirroring the old flow; this names the tail even when a pre-load edit also moved
    main()'s digest."""
    main, first_statement, digests = POST_LOAD[command]
    actual = statements_fingerprint_from(main, first_statement)
    assert reviewed_sources(digests, actual), (
        f"{command}: mflux main() changed after load(): review _core.run; {REVIEW_FIRST} {actual}"
    )


def test_mflux_main_is_a_reviewed_one() -> None:
    """Bug: mflux's main() gains a pre-load step (or Command.prepare) or changes what happens after
    load(), and the copy in z_image.py and _core.run quietly stop matching the plain command."""
    actual = ast_fingerprint(z_image_generate.main)
    assert reviewed_sources(MAIN_DIGESTS, actual), f"z_image_generate.main changed: {REVIEW_FIRST} {actual}"


def test_mflux_generate_image_is_a_reviewed_one() -> None:
    """Bug: mflux's ZImage.generate_image changes what it records or how it resolves guidance, the scheduler or
    generation_parameters, and FakeZImage.generate_image keeps building images the old way, so the sidecar tests here
    pass against a record the real command no longer writes."""
    actual = ast_fingerprint(ZImage.generate_image)
    assert reviewed_sources(GENERATE_IMAGE_DIGESTS, actual), (
        f"ZImage.generate_image changed: re-review FakeZImage.generate_image in tests/_fakes.py against it; "
        f"{REVIEW_FIRST} {actual}"
    )


def test_mflux_load_is_a_reviewed_one() -> None:
    """Bug: mflux's ZImageCommand.load starts doing more than building the model (another option applied to it, a
    different return), and the plugin, which calls load() directly, changes behaviour with no pin moving."""
    actual = ast_fingerprint(ZImageCommand.load)
    assert reviewed_sources(LOAD_DIGESTS, actual), f"ZImageCommand.load changed: {REVIEW_FIRST} {actual}"


def pinned_sources() -> dict[str, set[str]]:
    """Per pinned part, the labels of the reviewed mflux sources that hold its current value."""
    return {
        "build_parser": reviewed_sources(BUILD_PARSER_DIGESTS, ast_fingerprint(z_image_generate.build_parser)),
        "options": reviewed_sources(OPTIONS, parser_options()),
        "main": reviewed_sources(MAIN_DIGESTS, ast_fingerprint(z_image_generate.main)),
        "load": reviewed_sources(LOAD_DIGESTS, ast_fingerprint(ZImageCommand.load)),
        "generate_image": reviewed_sources(GENERATE_IMAGE_DIGESTS, ast_fingerprint(ZImage.generate_image)),
        "prediction step": reviewed_sources(PREDICT_DIGESTS, prediction_step_digests()),
        **{
            f"post-load {command}": reviewed_sources(digests, statements_fingerprint_from(main, first_statement))
            for command, (main, first_statement, digests) in POST_LOAD.items()
        },
    }


def test_every_pinned_part_comes_from_one_reviewed_source() -> None:
    """Bug: each pinned part is checked on its own against every label, so an mflux that pairs one source's parser
    with another source's generate_image (a combination nobody reviewed) passes."""
    found = pinned_sources()
    assert set.intersection(*found.values()), f"no reviewed mflux source holds all of these: {found}"


def _imported_mflux_is_the_installed_one() -> bool:
    """False when a source tree put first on PYTHONPATH shadows the installed mflux: its metadata then describes
    other code."""
    installed = Path(importlib.metadata.distribution("mflux").locate_file("mflux/__init__.py"))
    return mflux.__file__ is not None and installed.resolve() == Path(mflux.__file__).resolve()


@pytest.mark.skipif(not _imported_mflux_is_the_installed_one(), reason="the imported mflux is not the installed one")
def test_the_mflux_version_is_one_the_matched_source_reports() -> None:
    """Bug: an mflux release ships the code last reviewed on main, every pin matches the main label, and nothing asks
    for a label naming that release."""
    version = importlib.metadata.version("mflux")
    matched = sorted(set.intersection(*pinned_sources().values()))
    assert any(version in REPORTED_VERSIONS[source] for source in matched), (
        f"mflux {version} holds the values recorded under {matched}, a source that reports "
        f"{[REPORTED_VERSIONS[source] for source in matched]}: review mflux {version} and record these values under "
        f"a label for that release, with its version"
    )


def test_copied_preload_statements_match_mflux_main() -> None:
    """Bug: the copy in z_image.py drifts from mflux's main() (a dropped or changed warning branch),
    so the plugin no longer behaves like mflux-generate-z-image before load."""
    mflux_statements = set(function_body_dumps(z_image_generate.main))
    for copied in (z_image.warn_ineffective_options,):
        dumps = function_body_dumps(copied)
        # An empty body would match vacuously.
        assert dumps, f"{copied.__name__} has no statements to compare"
        unmatched = [dump for dump in dumps if dump not in mflux_statements]
        assert unmatched == [], f"{copied.__name__} no longer matches a statement of mflux's main()"


def test_command_steps_keep_the_signatures_the_plugin_calls() -> None:
    """Bug: mflux renames or reorders a ZImageCommand step parameter, and _core.run's positional
    generate(model, args, seed, prompt) call binds the wrong values."""
    assert list(inspect.signature(ZImageCommand.validate).parameters) == ["args"]
    assert list(inspect.signature(ZImageCommand.load).parameters) == ["args"]
    assert list(inspect.signature(ZImageCommand.generate).parameters) == ["model", "args", "seed", "prompt"]
    assert inspect.signature(ZImageCommand.validate).return_annotation is ModelConfig
    assert inspect.signature(ZImageCommand.generate).return_annotation is GeneratedImage


def _after_checks_on(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["mflux-generate-z-image-myplugin", "--prompt", "x", *argv])
    args = z_image.build_parser().parse_args()
    hook = z_image.ADAPTER.after_checks
    assert hook is not None
    hook(args, ZImageCommand.validate(args))


@pytest.mark.skipif(not HAS_DECODE_OPTIONS, reason="this mflux's Z-Image decode still tiles")
@pytest.mark.parametrize("argv", [["--vae-tiling"], ["--vae-tile-size", "256"], ["--vae-tile-size=256"]])
def test_vae_tiling_warns_before_load_where_the_decode_ignores_it(monkeypatch, argv) -> None:
    """Bug: the pre-load hook leaves out the VAEDecodeOptions.warn_encode_only_tiling() call that mflux's main() makes
    right after parsing, so on an mflux whose Z-Image decode ignores --vae-tiling the plugin runs it without the
    warning mflux-generate-z-image prints."""
    with pytest.warns(UserWarning, match="change only the encode of an input image"):
        _after_checks_on(monkeypatch, argv)


@pytest.mark.skipif(not HAS_DECODE_OPTIONS, reason="this mflux's Z-Image decode still tiles")
def test_vae_tiling_warning_stays_off_without_a_tiling_flag(monkeypatch) -> None:
    """Bug: the plugin warns that --vae-tiling changes only the encode whether or not the flag was given (a bare
    warnings.warn in place of mflux's warn_encode_only_tiling()), so every run on such an mflux gets a false warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _after_checks_on(monkeypatch, [])


@pytest.mark.skipif(not HAS_DECODE_OPTIONS, reason="this mflux's Z-Image decode still tiles")
def test_vae_tiling_warning_comes_before_the_model_loads(run_command, tmp_path: Path) -> None:
    """Bug: the tiling warning moves behind load() (a post-load hook, or _core.run calling after_checks after load),
    so the plugin prints it after the weights load, not before as mflux-generate-z-image does."""
    models_at_warning: list[int] = []

    def record(message: Warning | str, *args: Any, **kwargs: Any) -> None:
        if "change only the encode of an input image" in str(message):
            models_at_warning.append(len(FakeZImage.instances))

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = record
        run_command(
            ["--prompt", "x", "--seed", "1", "--steps", "5", "--vae-tiling", "--output", str(tmp_path / "o.png")]
        )
    assert models_at_warning == [0]
    assert len(FakeZImage.instances) == 1  # the run did load the model, after the warning


@pytest.mark.skipif(HAS_DECODE_OPTIONS, reason="this mflux's Z-Image decode ignores tiling")
@pytest.mark.parametrize("argv", [["--vae-tiling"], ["--vae-tile-size", "256"]])
def test_vae_tiling_stays_silent_where_the_decode_still_tiles(monkeypatch, argv) -> None:
    """Bug: the plugin warns that --vae-tiling changes only the encode on every mflux, so on the mflux 0.22.0 and
    0.23.0 releases, where the flag does tile the decode, it tells the user something false."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _after_checks_on(monkeypatch, argv)


def test_a_broken_decode_options_module_fails_the_import(monkeypatch) -> None:
    """Bug: the guarded import in z_image.py takes any ModuleNotFoundError (or ImportError) for an mflux without
    vae_decode_options, so where that module is there but cannot import its own dependency, the plugin runs
    without mflux's tiling warning instead of failing."""
    missing = "mflux.models.common.vae.block_decoder"
    real_import = builtins.__import__

    def broken_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mflux.models.common.vae.vae_decode_options":
            raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    # The reload stops at the guarded import; every name z_image held before keeps its value.
    with pytest.raises(ModuleNotFoundError) as caught:
        importlib.reload(z_image)
    assert caught.value.name == missing


@pytest.mark.skipif(not HAS_DECODE_OPTIONS, reason="this mflux has no --vae-full-precision")
@pytest.mark.parametrize(
    ("argv", "decode_mode", "recorded"), [([], "low_memory", None), (["--vae-full-precision"], "full_precision", True)]
)
def test_vae_full_precision_reaches_the_model_and_the_record(
    run_command, tmp_path: Path, argv, decode_mode, recorded
) -> None:
    """Bug: the plugin loads without mflux's VAEDecodeOptions.apply (or refuses or drops --vae-full-precision), so the
    decode runs in strips although the user asked for the full-image decode; or FakeZImage.generate_image leaves out
    the vae_full_precision key ZImage.generate_image records, and a -C replay of the sidecar loses the decode mode."""
    out = tmp_path / "out.png"
    run_command(["--prompt", "x", "--seed", "1", "--steps", "5", "--make-conf", "--output", str(out), *argv])
    (model,) = FakeZImage.instances
    assert model.vae.decode_mode == decode_mode
    assert json.loads(out.with_suffix(".metadata.json").read_text()).get("vae_full_precision") == recorded


def test_load_returns_the_class_the_fake_stands_in_for() -> None:
    """Bug: mflux's load() starts returning another class, and FakeZImage (built from ZImage's signatures) no longer
    stands in for what the command really loads."""
    assert inspect.signature(ZImageCommand.load).return_annotation is ZImage


def test_the_prediction_step_and_its_compile_gate_are_reviewed_ones() -> None:
    """Bug: mflux changes ZImage._predict (its factory signature, or when it wraps the step in mx.compile) or the chip
    check AppleSiliconUtil.is_m1_or_m2, and the compile rules in AGENTS.md, and any _predict a plugin installs,
    describe a step mflux no longer builds."""
    actual = prediction_step_digests()
    assert reviewed_sources(PREDICT_DIGESTS, actual), (
        f"ZImage._predict or AppleSiliconUtil.is_m1_or_m2 changed: re-read AGENTS.md's compile rules against them; "
        f"{REVIEW_FIRST} {actual}"
    )
