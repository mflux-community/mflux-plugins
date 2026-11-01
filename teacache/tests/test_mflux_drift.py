"""Pins the parts of mflux's Z-Image and Qwen-Image commands this package calls or copies.

Every pinned value is recorded once per mflux source it was reviewed against, under a label that names that source
(a release, or a main commit and the review date). A value no label holds fails. When one fails after an mflux
update: read mflux's diff of the command's file (src/mflux/models/z_image/cli/z_image_generate.py or
src/mflux/models/qwen/cli/qwen_image_generate.py), bring z_image.py or qwen_image.py (and _core.run, if the flow after
load() changed) in line with it, then add the values the failure messages print under a new label."""

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
from mflux.models.qwen.cli import qwen_image_generate
from mflux.models.qwen.cli.qwen_image_generate import QwenImageCommand
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.z_image.cli import z_image_generate
from mflux.models.z_image.cli.z_image_generate import ZImageCommand
from mflux.models.z_image.variants.z_image import ZImage
from mflux.utils.generated_image import GeneratedImage

import mflux
from mflux.extras.teacache import qwen_image, z_image

# mflux main a3b21fc added VAEDecodeOptions (the --vae-full-precision decode, and a warning that --vae-tiling no
# longer changes the decode); the mflux 0.22.0 / 0.23.0 releases have neither.
HAS_DECODE_OPTIONS = importlib.util.find_spec("mflux.models.common.vae.vae_decode_options") is not None

# The reviewed mflux sources. The released digests were recorded on PyPI 0.22.0 (commit 31c9640); v.0.23.0 has the
# same z_image_generate.py, variants/z_image.py and cli/parser/parsers.py. CI checks every digest on CPython 3.11 and
# 3.13.
RELEASED = "mflux v.0.22.0 / v.0.23.0 released"
# main a3b21fc: --vae-full-precision in build_parser, VAEDecodeOptions.warn_encode_only_tiling() in main() (mirrored
# by z_image.warn_before_load), VAEDecodeOptions.generation_parameters in ZImage.generate_image (mirrored by
# FakeZImage). ZImageCommand.load applies the decode mode itself, and the flow after load() is unchanged.
MAIN = "mflux main a3b21fc, reviewed 2026-10-23"
# PyPI 0.24.0 (tag v.0.24.0, commit 8a7fd0a), reviewed 2026-10-28 on CPython 3.13. Against main a3b21fc it leaves
# z_image_generate.py, variants/z_image.py and VAEDecodeOptions unchanged. Its parsers.py adds add_cfg_end_arguments
# (not called by the Z-Image parser), a bake_by_default parameter on add_lora_arguments (default True, as before) and
# a -C replay of bake_lora; none of it moves a pin here.
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

# The Qwen-Image command's parts, recorded on PyPI 0.22.0 and cross-checked on the 0.22.0 and 0.23.0 wheels, whose
# qwen_image_generate.py and variants/txt2img/qwen_image.py are the same. They have no value under MAIN yet: a part
# with no value under a label leaves that label open in the whole-set check (sources_not_ruled_out), and its own test
# still has to match a recorded value. QwenImage.generate_image's value is the one mlx-teacache records for mflux
# 0.22.0. PyPI 0.24.0, reviewed 2026-10-29 on CPython 3.13: both files are the same as in 0.23.0, and so are the
# parser helpers the Qwen-Image parser calls (add_lora_arguments keeps bake_by_default=True; it does not call
# add_cfg_end_arguments), so every Qwen-Image value is the released one.
QWEN_BUILD_PARSER_DIGESTS = {RELEASED: "9b3d21e5b122a1b9", RELEASED_0_24: "9b3d21e5b122a1b9"}
QWEN_MAIN_DIGESTS = {RELEASED: "74c1b41aa77a0ef0", RELEASED_0_24: "74c1b41aa77a0ef0"}
QWEN_VALIDATE_DIGESTS = {RELEASED: "7e09191dff32f856", RELEASED_0_24: "7e09191dff32f856"}
QWEN_LOAD_DIGESTS = {RELEASED: "b18636d16409e098", RELEASED_0_24: "b18636d16409e098"}
QWEN_GENERATE_DIGESTS = {RELEASED: "ee92a1a33b87793d", RELEASED_0_24: "ee92a1a33b87793d"}
QWEN_GENERATE_IMAGE_DIGESTS = {RELEASED: "9e6846e3f5ed09bb", RELEASED_0_24: "9e6846e3f5ed09bb"}
# Per pinned Qwen part: the function, its recorded values, and what to bring in line when it moves.
QWEN_PINS = {
    "qwen_image_generate.build_parser": (
        qwen_image_generate.build_parser,
        QWEN_BUILD_PARSER_DIGESTS,
        "qwen_image.build_parser and the refusal tests (an option that skips or caches steps needs a refusal)",
    ),
    "qwen_image_generate.main": (
        qwen_image_generate.main,
        QWEN_MAIN_DIGESTS,
        "a new pre-load step in main() needs an after_checks hook in qwen_image.py",
    ),
    "QwenImageCommand.validate": (
        QwenImageCommand.validate,
        QWEN_VALIDATE_DIGESTS,
        "qwen_image.validate, which wraps it and refuses the original checkpoint first",
    ),
    "QwenImageCommand.load": (
        QwenImageCommand.load,
        QWEN_LOAD_DIGESTS,
        "_core.run, which calls load() directly and applies TeaCache to what it returns",
    ),
    "QwenImageCommand.generate": (
        QwenImageCommand.generate,
        QWEN_GENERATE_DIGESTS,
        "_core.run, which passes it the parsed arguments and relies on it resolving --width and --height",
    ),
    "QwenImage.generate_image": (
        QwenImage.generate_image,
        QWEN_GENERATE_IMAGE_DIGESTS,
        "FakeQwenImage.generate_image and LoopingFakeQwenImage.denoise in tests/_fakes.py",
    ),
}
# Every option string and dest of the Qwen-Image parser, sorted. Against the Z-Image tuples above it lacks
# --compute-precision, --float32 and --no-float32 (dests compute_precision and float32) and adds nothing.
QWEN_OPTION_STRINGS = (
    "--auto-seeds", "--bake-lora", "--base-model", "--battery-percentage-stop-limit", "--config-from-conf",
    "--config-from-metadata", "--guidance", "--height", "--help", "--image", "--image-path", "--image-strength",
    "--lora", "--lora-paths", "--lora-scales", "--lora-style", "--low-ram", "--make-conf", "--metadata",
    "--mlx-cache-limit-gb", "--model", "--negative-prompt", "--no-bake-lora", "--no-exif", "--no-metadata",
    "--no-pid-decode", "--output", "--pid-decode", "--pid-degrade-sigma", "--prompt", "--prompt-file", "--quantize",
    "--scheduler", "--seed", "--steps", "--stepwise-image-output-dir", "--vae-tile-size", "--vae-tiling", "--verbose",
    "--width", "-B", "-C", "-h", "-m", "-q", "-v",
)  # fmt: skip
QWEN_DESTS = (
    "auto_seeds", "bake_lora", "base_model", "battery_percentage_stop_limit", "config_from_metadata", "guidance",
    "height", "help", "image", "image_path", "image_strength", "lora", "lora_paths", "lora_scales", "lora_style",
    "low_ram", "metadata", "mlx_cache_limit_gb", "model", "negative_prompt", "no_metadata", "output", "pid_decode",
    "pid_degrade_sigma", "prompt", "prompt_file", "quantize", "scheduler", "seed", "steps",
    "stepwise_image_output_dir", "vae_tile_size", "vae_tiling", "verbose", "width",
)  # fmt: skip
QWEN_OPTIONS = {RELEASED: (QWEN_OPTION_STRINGS, QWEN_DESTS), RELEASED_0_24: (QWEN_OPTION_STRINGS, QWEN_DESTS)}

# Per adapter: the mirrored mflux main(), its first statement after the pre-load block, and the digest of everything
# from there on (load, callback registration, the per-seed generate/save loop, the except/finally). _core.run mirrors
# that tail; a new adapter adds its row here.
POST_LOAD = {
    "mflux-generate-z-image-teacache": (
        z_image_generate.main,
        "model = ZImageCommand.load(args)",
        {RELEASED: "812cb2e320c4eaee", MAIN: "812cb2e320c4eaee", RELEASED_0_24: "812cb2e320c4eaee"},
    ),
    # mflux's Qwen main() resolves the sizes into a copy of args before its seed loop; _core.run passes the parsed
    # args, which changes no output because QwenImageCommand.generate resolves them again.
    "mflux-generate-qwen-teacache": (
        qwen_image_generate.main,
        "model = QwenImageCommand.load(args)",
        {RELEASED: "ff216a1e1a70ec70", RELEASED_0_24: "ff216a1e1a70ec70"},
    ),
}

REVIEW_FIRST = (
    "review the mflux change first (AGENTS.md, 'When the drift test fails'), then record this value under a label "
    "naming the mflux source you reviewed:"
)


def reviewed_sources(accepted: Mapping[str, object], actual: object) -> set[str]:
    """The labels of the reviewed mflux sources whose recorded value is `actual`; empty for an unreviewed one."""
    return {source for source, value in accepted.items() if value == actual}


def sources_not_ruled_out(accepted: Mapping[str, object], actual: object, labels: tuple[str, ...]) -> set[str]:
    """The labels `actual` leaves possible for the whole set: those under which it is the recorded value, plus those
    under which this part has no value yet (a command added since that source was last reviewed). A part's own test
    still requires a recorded value."""
    return reviewed_sources(accepted, actual) | {label for label in labels if label not in accepted}


def parser_options(module: Any = z_image_generate) -> tuple[tuple[str, ...], tuple[str, ...]]:
    parser = module.build_parser()
    strings = tuple(sorted(option for action in parser._actions for option in action.option_strings))
    dests = tuple(sorted({action.dest for action in parser._actions}))
    return strings, dests


def test_mflux_build_parser_is_a_reviewed_one() -> None:
    """Bug: mflux's Z-Image parser gains or drops an option (a step-cache flag, say) and the plugin
    keeps passing through a parser nobody checked against its refusals."""
    actual = ast_fingerprint(z_image_generate.build_parser)
    assert reviewed_sources(BUILD_PARSER_DIGESTS, actual), (
        f"z_image_generate.build_parser changed: {REVIEW_FIRST} {actual}"
    )


def test_mflux_parser_options_are_reviewed_ones() -> None:
    """Bug: mflux adds an option through a shared parser helper (a step-cache or TeaCache-style flag on Z-Image, say);
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


@pytest.mark.parametrize("part", sorted(QWEN_PINS))
def test_mflux_qwen_part_is_a_reviewed_one(part: str) -> None:
    """Bug: mflux changes a part of its Qwen-Image command that the plugin calls, wraps or reproduces (a parser
    option, a pre-load step in main(), what validate accepts, what load builds, the size resolve in generate, what
    generate_image records), and the plugin keeps running it unreviewed."""
    function, digests, review = QWEN_PINS[part]
    actual = ast_fingerprint(function)
    assert reviewed_sources(digests, actual), f"{part} changed: bring {review} in line; {REVIEW_FIRST} {actual}"


def test_mflux_qwen_parser_options_are_reviewed_ones() -> None:
    """Bug: mflux adds an option to the Qwen-Image parser through a shared helper (a step-cache flag, say);
    build_parser's own digest does not move, and the plugin passes the flag through without a refusal."""
    strings, dests = parser_options(qwen_image_generate)
    assert reviewed_sources(QWEN_OPTIONS, (strings, dests)), (
        f"qwen_image_generate.build_parser() options changed: {REVIEW_FIRST}\n{strings}\n{dests}"
    )


def test_a_part_with_no_value_under_a_label_leaves_that_label_open() -> None:
    """Bug: the whole-set check reads a part recorded under the release only (a command added since mflux main was
    last reviewed) as ruling main out, so on mflux main it fails although every part's own pin passes; or it leaves a
    label open where the part has a different value there, so a mix of two sources passes."""
    labels = ("release", "main")
    assert sources_not_ruled_out({"release": "a"}, "a", labels) == {"release", "main"}
    assert sources_not_ruled_out({"release": "a"}, "b", labels) == {"main"}
    assert sources_not_ruled_out({"release": "a", "main": "b"}, "a", labels) == {"release"}
    assert sources_not_ruled_out({"release": "a", "main": "b"}, "c", labels) == set()


def pinned_sources(*, recorded_only: bool = False) -> dict[str, set[str]]:
    """Per pinned part, the labels of the reviewed mflux sources it leaves possible (sources_not_ruled_out), or with
    recorded_only those under which its value is recorded (reviewed_sources)."""
    labels = tuple(REPORTED_VERSIONS)

    def possible(accepted: Mapping[str, object], actual: object) -> set[str]:
        if recorded_only:
            return reviewed_sources(accepted, actual)
        return sources_not_ruled_out(accepted, actual, labels)

    return {
        "build_parser": possible(BUILD_PARSER_DIGESTS, ast_fingerprint(z_image_generate.build_parser)),
        "options": possible(OPTIONS, parser_options()),
        "main": possible(MAIN_DIGESTS, ast_fingerprint(z_image_generate.main)),
        "load": possible(LOAD_DIGESTS, ast_fingerprint(ZImageCommand.load)),
        "generate_image": possible(GENERATE_IMAGE_DIGESTS, ast_fingerprint(ZImage.generate_image)),
        "qwen options": possible(QWEN_OPTIONS, parser_options(qwen_image_generate)),
        **{part: possible(digests, ast_fingerprint(function)) for part, (function, digests, _) in QWEN_PINS.items()},
        **{
            f"post-load {command}": possible(digests, statements_fingerprint_from(main, first_statement))
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
    for a label naming that release; or a part has no value under that release's label (the Qwen-Image command left
    out of its review), and the release passes for that command unreviewed."""
    version = importlib.metadata.version("mflux")
    matched = sorted(set.intersection(*pinned_sources(recorded_only=True).values()))
    assert any(version in REPORTED_VERSIONS[source] for source in matched), (
        f"mflux {version} holds the values recorded under {matched}, a source that reports "
        f"{[REPORTED_VERSIONS[source] for source in matched]}: review mflux {version} and record every part's value "
        f"under a label for that release, with its version"
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


def test_qwen_command_steps_keep_the_signatures_the_plugin_calls() -> None:
    """Bug: mflux renames or reorders a QwenImageCommand step parameter, and _core.run's positional
    generate(model, args, seed, prompt) call (or qwen_image.validate's call) binds the wrong values."""
    assert list(inspect.signature(QwenImageCommand.validate).parameters) == ["args"]
    assert list(inspect.signature(QwenImageCommand.load).parameters) == ["args"]
    assert list(inspect.signature(QwenImageCommand.generate).parameters) == ["model", "args", "seed", "prompt"]
    assert inspect.signature(QwenImageCommand.validate).return_annotation is ModelConfig
    assert inspect.signature(QwenImageCommand.generate).return_annotation is GeneratedImage


def _after_checks_on(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["mflux-generate-z-image-teacache", "--prompt", "x", *argv])
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


def test_load_returns_the_pipeline_the_adapter_names() -> None:
    """Bug: mflux's load() starts returning another class, and match_variant is asked about the wrong
    pipeline, so the pre-load check accepts or refuses the wrong models."""
    assert inspect.signature(ZImageCommand.load).return_annotation is ZImage
    assert z_image.PIPELINE is ZImage
    assert z_image.ADAPTER.pipeline is ZImage


def test_qwen_load_returns_the_pipeline_the_adapter_names() -> None:
    """Bug: mflux's QwenImageCommand.load starts returning another class, or the Qwen adapter names another pipeline,
    so match_variant is asked about the wrong class and the pre-load check accepts or refuses the wrong models."""
    assert inspect.signature(QwenImageCommand.load).return_annotation is QwenImage
    assert qwen_image.PIPELINE is QwenImage
    assert qwen_image.ADAPTER.pipeline is QwenImage
    assert qwen_image.ADAPTER.load is QwenImageCommand.load
    assert qwen_image.ADAPTER.generate is QwenImageCommand.generate
