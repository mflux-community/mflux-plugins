"""Pins the parts of mflux's Z-Image command this package calls or copies.

Every pinned value is recorded once per mflux source it was reviewed against, under a label that names that source
(a release, or a main commit and the review date). A value no label holds fails. When one fails after an mflux
update: read mflux's diff of src/mflux/models/z_image/cli/z_image_generate.py, bring z_image.py (and _core.run, if
the flow after load() changed) in line with it, then add the values the failure messages print under a new label."""

import ast
import importlib.metadata
import inspect
import json
import sys
import warnings
from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from _fakes import FakeZImage
from _fingerprint import ast_fingerprint, function_body_dumps, statements_fingerprint_from
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.z_image.cli import z_image_generate
from mflux.models.z_image.cli.z_image_generate import ZImageCommand
from mflux.models.z_image.variants.z_image import ZImage
from mflux.utils.apple_silicon import AppleSiliconUtil
from mflux.utils.generated_image import GeneratedImage

import mflux
from mflux.extras.myplugin import _core, z_image

# The reviewed mflux source: PyPI 0.24.0 (tag v.0.24.0, commit 8a7fd0a), the oldest release this package supports,
# reviewed on 2026-10-28 on CPython 3.13. CI checks every digest on CPython 3.11 and 3.13.
RELEASED_0_24 = "mflux v.0.24.0 released"
# The versions each reviewed source reports.
REPORTED_VERSIONS = {RELEASED_0_24: ("0.24.0",)}

BUILD_PARSER_DIGESTS = {RELEASED_0_24: "6a85a43cc52fd44f"}
MAIN_DIGESTS = {RELEASED_0_24: "1f3058eda927d0fe"}
# ZImageCommand.load, which _core.run calls for the model: it builds the ZImage, calls VAEDecodeOptions.apply(args,
# model) and returns the model.
LOAD_DIGESTS = {RELEASED_0_24: "d91dfaa735845a7b"}
# ZImage.generate_image, the method tests/_fakes.py's FakeZImage.generate_image reproduces up to the image record.
GENERATE_IMAGE_DIGESTS = {RELEASED_0_24: "88204535dc85b78e"}
# ZImage._predict, the factory a plugin replaces to keep per-step state, and the chip check that decides whether mflux
# compiles the step it returns. AGENTS.md's "Compile and memory" section describes both.
PREDICT_DIGESTS = {RELEASED_0_24: ("016c64c92ceefbdf", "3264a79fbfa0f1a5")}

# Every option string and dest of the Z-Image parser, sorted. build_parser's digest sees only the helper calls it
# makes; an option added inside a shared helper (add_image_generator_arguments and the like) shows up only here.
OPTION_STRINGS = (
    "--auto-seeds", "--bake-lora", "--base-model", "--battery-percentage-stop-limit", "--compute-precision",
    "--config-from-conf", "--config-from-metadata", "--float32", "--guidance", "--height", "--help", "--image",
    "--image-path", "--image-strength", "--lora", "--lora-paths", "--lora-scales", "--lora-style", "--low-ram",
    "--make-conf", "--metadata", "--mlx-cache-limit-gb", "--model", "--negative-prompt", "--no-bake-lora",
    "--no-exif", "--no-float32", "--no-metadata", "--no-pid-decode", "--no-vae-full-precision", "--output",
    "--pid-decode", "--pid-degrade-sigma", "--prompt", "--prompt-file", "--quantize", "--scheduler", "--seed",
    "--steps", "--stepwise-image-output-dir", "--vae-full-precision", "--vae-tile-size", "--vae-tiling", "--verbose",
    "--width", "-B", "-C", "-h", "-m", "-q", "-v",
)  # fmt: skip
DESTS = (
    "auto_seeds", "bake_lora", "base_model", "battery_percentage_stop_limit", "compute_precision",
    "config_from_metadata", "float32", "guidance", "height", "help", "image", "image_path", "image_strength", "lora",
    "lora_paths", "lora_scales", "lora_style", "low_ram", "metadata", "mlx_cache_limit_gb", "model", "negative_prompt",
    "no_metadata", "output", "pid_decode", "pid_degrade_sigma", "prompt", "prompt_file", "quantize", "scheduler",
    "seed", "steps", "stepwise_image_output_dir", "vae_full_precision", "vae_tile_size", "vae_tiling", "verbose",
    "width",
)  # fmt: skip
OPTIONS = {RELEASED_0_24: (OPTION_STRINGS, DESTS)}

# Per adapter: the mirrored mflux main(), its first statement after the pre-load block, and the digest of everything
# from there on (load, callback registration, the per-seed generate/save loop, the except/finally). _core.run mirrors
# that tail; a new adapter adds its row here.
POST_LOAD = {
    "mflux-generate-z-image-myplugin": (
        z_image_generate.main,
        "model = ZImageCommand.load(args)",
        {RELEASED_0_24: "812cb2e320c4eaee"},
    ),
}

# mflux's model-path rules, as (priority, name, check, action value). _core._refuse_unknown_model refuses before load
# the names these rules would send to the "error" rule inside load(): no path that exists and not org/model in shape
# (_core._is_hf_format, a copy of the predicate behind the "is_hf_format" check).
PATH_RESOLUTION_RULES = {
    RELEASED_0_24: [
        (0, "none", "is_none", "Use local filesystem path"),
        (1, "local", "exists_locally", "Use local filesystem path"),
        (2, "hf_cached", "is_hf_cached", "Use cached HuggingFace files (no network)"),
        (3, "hf_download", "is_hf_format", "Download from HuggingFace"),
        (4, "error", "always", "Path not found"),
    ]
}
# PathResolution._check and _execute, which decide what each rule above tests and does. _core.REVIEWED_PATH_RULES
# names the rules, not what they do; these two pin that.
PATH_RESOLUTION_CHECK_DIGESTS = {RELEASED_0_24: "b0f9ed4e22018374"}
PATH_RESOLUTION_EXECUTE_DIGESTS = {RELEASED_0_24: "c7fd6091bc7ea323"}

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


def parser_options() -> tuple[tuple[str, ...], tuple[str, ...]]:
    parser = z_image_generate.build_parser()
    strings = tuple(sorted(option for action in parser._actions for option in action.option_strings))
    dests = tuple(sorted({action.dest for action in parser._actions}))
    return strings, dests


def prediction_step_digests() -> tuple[str, str]:
    return ast_fingerprint(ZImage._predict), ast_fingerprint(AppleSiliconUtil.is_m1_or_m2)


def path_resolution_rules() -> list[tuple[int, str, str, str]]:
    return sorted((rule.priority, rule.name, rule.check, rule.action.value) for rule in PathResolution.RULES)


def test_mflux_path_resolution_rules_are_reviewed_ones() -> None:
    """Bug: mflux adds or reorders a model-path rule (a registry lookup before "error", say), and the plugin's
    refusal of names that are no path and not org/model in shape turns away a --model mflux now loads."""
    actual = path_resolution_rules()
    assert reviewed_sources(PATH_RESOLUTION_RULES, actual), (
        f"PathResolution.RULES changed: review _core._refuse_unknown_model; {REVIEW_FIRST} {actual}"
    )


@pytest.mark.parametrize(
    ("part", "function", "digests"),
    [
        ("PathResolution._check", PathResolution._check, PATH_RESOLUTION_CHECK_DIGESTS),
        ("PathResolution._execute", PathResolution._execute, PATH_RESOLUTION_EXECUTE_DIGESTS),
    ],
)
def test_mflux_path_rule_dispatch_is_a_reviewed_one(part: str, function, digests: dict[str, str]) -> None:
    """Bug: mflux keeps the rule set but changes what a check tests or an action does (exists_locally rejects an
    empty folder, LOCAL starts to raise for ""), and the refusal keeps answering for the old behaviour."""
    actual = ast_fingerprint(function)
    assert reviewed_sources(digests, actual), (
        f"{part} changed: review _core._refuse_unknown_model; {REVIEW_FIRST} {actual}"
    )


# Names no download rule can take (none is org/model in shape), with whether mflux 0.24.0 raises FileNotFoundError
# for each in a folder that holds ./weights/ and ./one.safetensors, with HOME there too.
NAMES_BEFORE_THE_DOWNLOAD_RULE = [
    ("", False), ("weights", False), ("./weights", False), ("~/weights", False), ("one.safetensors", False),
    ("flux-dev", True), ("./no-such-folder", True), ("../no-such-folder", True), ("~/no-such-folder", True),
    ("/no/such/folder", True), ("no/such/folder", True), ("./org", True), ("~/org", True),
]  # fmt: skip


def test_mflux_resolves_an_empty_model_name_to_no_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Bug: mflux starts to treat --model "" as a path (or an error), and the plugin, which lets "" through to run the
    command's default model, sends it into a load that fails or loads the working folder."""
    monkeypatch.chdir(tmp_path)
    assert PathResolution.resolve("") is None


@pytest.mark.parametrize(("name", "raises"), NAMES_BEFORE_THE_DOWNLOAD_RULE)
def test_the_plugin_refuses_exactly_the_names_mflux_cannot_resolve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str, raises: bool
) -> None:
    """Bug: the refusal and mflux's own resolve part ways on a name that never reaches a download (the refusal skips
    expanding ~, or mflux starts to resolve a bare name), so the plugin refuses a model mflux loads or lets through a
    name that ends in load()'s traceback."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "weights").mkdir()
    (tmp_path / "one.safetensors").touch()
    assert not PathResolution._is_hf_format(name), "the name would reach the download rule"
    try:
        PathResolution.resolve(name)
        mflux_raises = False
    except FileNotFoundError:
        mflux_raises = True
    try:
        _core._refuse_unknown_model(z_image.build_parser(), Namespace(model_path=name))
        refused = False
    except SystemExit:
        refused = True
    assert (refused, mflux_raises) == (raises, raises)


def test_copied_hf_format_check_matches_mflux() -> None:
    """Bug: _core._is_hf_format drifts from mflux's PathResolution._is_hf_format (a check dropped or changed in the
    copy, or mflux changes its own), so the plugin refuses a repo id mflux downloads or lets through a name mflux
    rejects inside load()."""
    assert function_body_dumps(_core._is_hf_format) == function_body_dumps(PathResolution._is_hf_format)


@pytest.mark.parametrize(
    "name",
    [
        # The names tests/test_run.py's refusal tests use: refused, a local path, a repo id not on this machine.
        "flux-dev", "./no-such-folder", "../no-such-folder", "~/no-such-folder", "/no/such/folder", "no/such/folder",
        "weights", "./weights", "~/weights", "one.safetensors", "someorg/not-downloaded",
        # org/model names, and near misses.
        "Tongyi-MAI/Z-Image", "black-forest-labs/FLUX.1-dev", "org/model", "org/", "/model", "./org", "../org", "~/org", "",
        None,
    ],
)  # fmt: skip
def test_copied_hf_format_check_agrees_with_mflux(name: str | None) -> None:
    """Bug: the copy answers differently from mflux's predicate for a name the refusal sees (a ./ prefix not
    excluded, two slashes taken as a repo id), so the plugin and mflux disagree on what is a Hugging Face repo id."""
    assert _core._is_hf_format(name) == PathResolution._is_hf_format(name)


def test_copied_argv_scan_matches_mflux() -> None:
    """Bug: _core._option_was_provided drifts from mflux's CommandLineParser._option_was_provided (the --option=value
    or -qVALUE form dropped from the copy, or mflux changes its own), so the Z-Image negative-prompt warning fires
    where mflux's would not, or stays silent where it would fire."""
    assert ast_fingerprint(_core._option_was_provided) == ast_fingerprint(CommandLineParser._option_was_provided)


@pytest.mark.parametrize(
    ("argv", "names", "provided"),
    [
        (["--negative-prompt", "x"], ("--negative-prompt",), True),
        (["--negative-prompt=x"], ("--negative-prompt",), True),
        (["--negative-promptx"], ("--negative-prompt",), False),
        (["--prompt", "use --negative-prompt"], ("--negative-prompt",), False),
        ([], ("--negative-prompt",), False),
        (["-q", "8"], ("-q",), True),
        (["-q8"], ("-q",), True),
        (["-"], ("-q",), False),
        (["--quantize=8"], ("-q", "--quantize"), True),
        (["--lora-scales", "0.5"], ("--lora-paths", "--lora-scales"), True),
        (["--guidance", "4"], ("--negative-prompt",), False),
    ],
)
def test_copied_argv_scan_agrees_with_mflux(monkeypatch: pytest.MonkeyPatch, argv, names, provided: bool) -> None:
    """Bug: the copy answers differently from mflux's scan for an argv the Z-Image command sees (the --option=value
    form missed, a prefix taken as the option), so the plugin and mflux disagree on whether an option was given."""
    monkeypatch.setattr(sys, "argv", ["mflux-generate-z-image-myplugin", *argv])
    assert _core._option_was_provided(*names) == CommandLineParser._option_was_provided(*names) == provided


def _is_private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _is_mflux_module(module: str) -> bool:
    return module == "mflux" or module.startswith("mflux.")


def private_names(source: str) -> list[str]:
    """Every private name the source could reach mflux through: any non-dunder _ attribute (on any object, since the
    scan can't tell an mflux object from another), getattr/hasattr/setattr/delattr with a constant _ name, a _ name
    imported from mflux, and a _ part in an mflux module path. The package's own helpers are bare names imported
    relatively, so they don't count."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and _is_private(node.attr):
            found.append(f".{node.attr}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "hasattr", "setattr", "delattr"}
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
            and _is_private(node.args[1].value)
        ):
            found.append(f"{node.func.id}(..., {node.args[1].value!r})")
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and _is_mflux_module(node.module or ""):
            found += [f"from {node.module}" for part in (node.module or "").split(".") if _is_private(part)]
            found += [f"import {alias.name}" for alias in node.names if _is_private(alias.name)]
        elif isinstance(node, ast.Import):
            found += [
                f"import {alias.name}"
                for alias in node.names
                if _is_mflux_module(alias.name) and any(_is_private(part) for part in alias.name.split("."))
            ]
    return found


@pytest.mark.parametrize(
    ("source", "private"),
    [
        ("from mflux.cli.parser.parsers import CommandLineParser\nCommandLineParser._option_was_provided('-m')", True),
        ("from mflux.models.z_image.cli import z_image_generate\nz_image_generate.CommandLineParser._x()", True),
        ("parser = build_parser()\nparser._actions", True),
        ("from mflux.cli.parser.parsers import CommandLineParser as P\nP._option_was_provided('-m')", True),
        ("getattr(PathResolution, '_is_hf_format')", True),
        ("from mflux.cli.parser.parsers import _helper", True),
        ("from mflux.models.common._private import Name", True),
        ("import mflux.models._private", True),
        ("x.__name__", False),
        ("getattr(args, 'model_path', None)", False),
        ("getattr(args, '__doc__')", False),
        ("from mflux.cli.parser.parsers import CommandLineParser", False),
        ("from ._core import _option_was_provided\n_option_was_provided('-m')", False),
        ("import mflux.models.common.resolution.path_resolution", False),
    ],
)
def test_the_private_name_scan_sees_each_shape(source: str, private: bool) -> None:
    """Bug: the scan misses a shape of private use (a module-then-class chain, an instance, an alias, getattr with a
    constant name, a _ part in an mflux module path), so shipped code uses that private name and the scan below stays
    green; or it flags a dunder or the package's own private helpers, and the scan can never pass."""
    assert bool(private_names(source)) == private


def test_shipped_code_uses_no_private_mflux_name() -> None:
    """Bug: the package calls a private mflux name again (PathResolution._is_hf_format, say); with no upper bound on
    mflux, a rename there turns every run that reaches the call into a traceback."""
    paths = sorted(Path(_core.__file__).parent.rglob("*.py"))
    assert "_core.py" in {path.name for path in paths}, "the scan found no shipped code to read"
    found = [f"{path.name} {name}" for path in paths for name in private_names(path.read_text(encoding="utf-8"))]
    assert found == []


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
        "prediction step": possible(PREDICT_DIGESTS, prediction_step_digests()),
        "path resolution rules": possible(PATH_RESOLUTION_RULES, path_resolution_rules()),
        "PathResolution._check": possible(PATH_RESOLUTION_CHECK_DIGESTS, ast_fingerprint(PathResolution._check)),
        "PathResolution._execute": possible(PATH_RESOLUTION_EXECUTE_DIGESTS, ast_fingerprint(PathResolution._execute)),
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
    for a label naming that release; or a part has no value under that release's label (a command left out of its
    review), and the release passes for that command unreviewed."""
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
    # The one exemption: z_image.py calls _core's copy of the argv scan (test_copied_argv_scan_matches_mflux) where
    # mflux's main() calls CommandLineParser._option_was_provided. Spelled back to mflux's call before comparing.
    local, mflux_call = (
        "(Name,id='_option_was_provided')",
        "(Attribute,value=(Name,id='CommandLineParser'),attr='_option_was_provided')",
    )
    for copied in (z_image.warn_ineffective_options,):
        dumps = [dump.replace(local, mflux_call) for dump in function_body_dumps(copied)]
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


@pytest.mark.parametrize("argv", [["--vae-tiling"], ["--vae-tile-size", "256"], ["--vae-tile-size=256"]])
def test_vae_tiling_warns_before_load(monkeypatch, argv) -> None:
    """Bug: the pre-load hook leaves out the VAEDecodeOptions.warn_encode_only_tiling() call that mflux's main() makes
    right after parsing, so the plugin runs --vae-tiling, which the Z-Image decode ignores, without the warning
    mflux-generate-z-image prints."""
    with pytest.warns(UserWarning, match="change only the encode of an input image"):
        _after_checks_on(monkeypatch, argv)


def test_vae_tiling_warning_stays_off_without_a_tiling_flag(monkeypatch) -> None:
    """Bug: the plugin warns that --vae-tiling changes only the encode whether or not the flag was given (a bare
    warnings.warn in place of mflux's warn_encode_only_tiling()), so every run gets a false warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _after_checks_on(monkeypatch, [])


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
