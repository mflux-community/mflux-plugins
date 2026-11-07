"""The command flow every mflux-teacache command shares.

Each model family has a small adapter module (z_image.py) that names mflux's own command steps.
run() parses with mflux's parser, refuses anything TeaCache can't run before any weights load, then
loads with mflux, applies TeaCache, and generates, records and saves the way the plain mflux command
does."""

import os
import sys
import traceback
import warnings
from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import AVAILABLE_MODELS, ModelConfig
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.utils.exceptions import ModelConfigError, PromptFileReadError, StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.prompt_util import PromptUtil
from mlx_teacache import (
    InvalidStepWindowError,
    TeaCacheHandle,
    TeaCacheUncalibratedCheckpointWarning,
    TeaCacheValueError,
    VariantInfo,
    __version__ as mlx_teacache_version,
    apply_teacache,
    check_step_window,
    match_variant,
)

from . import __version__ as plugin_version

THRESHOLD_FLAG = "--teacache-threshold"
# mflux's --compute-precision (float16 only, as of mflux 0.24): any non-default value is allowed for now, with this one
# warning. Kept in one place so a later switch to a refusal is a small change.
COMPUTE_PRECISION_WARNING = (
    "TeaCache's skip decisions and image quality have not been measured with --compute-precision {value}; "
    "the run uses it as asked."
)


@dataclass(frozen=True)
class Adapter:
    """One mflux command as run() needs it.

    The required fields are mflux's own command steps. The hooks are optional, so an adapter names
    only what its mirrored command does around them:
    - before_validate(args): right after parsing and the threshold check, before validate (no current
      adapter uses it).
    - after_checks(args, model_config): after TeaCache's refusals, before load (Z-Image: warn_before_load,
      mflux's tiling, guidance and negative-prompt warnings), so a refused run prints one message.
    - active_steps(args, model_config): the denoising steps mflux will run; None uses
      default_active_steps (mflux's own Config.init_time_step).
    - calibrated_copies: repo ids known to hold the calibrated checkpoint's model at other precisions (copies
      whose model card declares it as base_model); --model set to one of them, in any case, does not warn.
      TeaCache's settings were measured on one build only, so these run unmeasured, just without the warning."""

    command: str
    plain_command: str
    pipeline: type
    family_keys: tuple[str, ...]
    build_parser: Callable[[], CommandLineParser]
    validate: Callable[[Namespace], ModelConfig]
    load: Callable[[Namespace], Any]
    generate: Callable[[Any, Namespace, int, str], GeneratedImage]
    latent_creator: type
    before_validate: Callable[[Namespace], None] | None = None
    after_checks: Callable[[Namespace, ModelConfig], None] | None = None
    active_steps: Callable[[Namespace, ModelConfig], int] | None = None
    calibrated_copies: tuple[str, ...] = ()


def add_teacache_arguments(parser: CommandLineParser) -> None:
    group = parser.add_argument_group("TeaCache")
    group.add_argument(
        THRESHOLD_FLAG,
        dest="teacache_threshold",
        type=float,
        default=None,
        metavar="VALUE",
        help=(
            "How much change TeaCache tolerates before it computes a step again: greater than 0, at most 1. "
            "Higher skips more steps and changes the image more. Default: the model's calibrated value."
        ),
    )


def default_active_steps(args: Namespace, model_config: ModelConfig) -> int:
    """The denoising steps mflux will run: --steps minus mflux's own init_time_step (an image-to-image
    run starts later; strength 1.0 leaves none). Loads no weights and opens no image: width and height
    keep Config's defaults."""
    init_time_step = Config(
        model_config=model_config,
        num_inference_steps=args.steps,
        image_path=getattr(args, "image_path", None),
        image_strength=getattr(args, "image_strength", None),
    ).init_time_step
    return args.steps - init_time_step


def supported_model_names(adapter: Adapter) -> tuple[str, ...]:
    """The --model names this command can run under TeaCache: every mflux alias of a family entry mlx-teacache's own
    registry accepts. A variant's model_names can be fewer (Qwen-Image-2512's are two of its four aliases)."""
    names: list[str] = []
    for key in adapter.family_keys:
        config = AVAILABLE_MODELS[key]
        if match_variant(config, adapter.pipeline) is not None:
            names.extend(name for name in config.aliases if name not in names)
    return tuple(names)


def teacache_metadata(handle: TeaCacheHandle, committed_before: int) -> dict[str, float | int | str] | None:
    """The image's TeaCache record, or None when this generation was not committed (interrupted, or
    the handle was restored mid-run). Flat keys and JSON primitives, like mflux's own step-cache key,
    because mflux writes them into the PNG and the JSON sidecar as they are."""
    stats = handle.stats
    last = stats.last_generation
    if stats.generations <= committed_before or last is None:
        return None
    record: dict[str, float | int | str] = {
        "teacache_threshold": float(handle.rel_l1_thresh),
        "teacache_skipped_steps": sum(1 for step in last.decisions if step.decision == "skipped"),
        "teacache_active_steps": int(last.num_steps),
        "teacache_variant": str(getattr(handle, "variant_id", "unknown")),
        "teacache_version": str(mlx_teacache_version),
        "teacache_decision_rule_revision": int(handle.decision_rule_revision),
        "teacache_plugin_version": str(plugin_version),
    }
    # None when mflux's package metadata is missing (a source tree on the path, say).
    if handle.mflux_version is not None:
        record["teacache_mflux_version"] = str(handle.mflux_version)
    return record


def _record_teacache(image: GeneratedImage, handle: TeaCacheHandle, committed_before: int) -> None:
    recorded = teacache_metadata(handle, committed_before)
    if recorded is None:
        return
    # Merged into what mflux recorded there (Z-Image: float32, compute_precision, which -C replays), and a new
    # dict: never mutate one the image may share (GeneratedImage.get_right_half shares it). A later family's image may
    # carry None there.
    image.generation_parameters = {**(image.generation_parameters or {}), **recorded}
    print(
        f"TeaCache: skipped {recorded['teacache_skipped_steps']} of {recorded['teacache_active_steps']} steps "
        f"(threshold {recorded['teacache_threshold']:g})"
    )


def _checked_threshold(parser: CommandLineParser, adapter: Adapter, value: float | None) -> float | None:
    if value is not None and not 0.0 < value <= 1.0:  # NaN fails both comparisons and is refused
        parser.error(
            f"{THRESHOLD_FLAG} must be greater than 0 and at most 1, got {value}. "
            f"Run {adapter.plain_command} to generate without TeaCache."
        )
    return value


# Copied from mflux's PathResolution._is_hf_format (src/mflux/models/common/resolution/path_resolution.py), the check
# its model-path rules use for a Hugging Face repo id. A copy, so a rename of that private method in a newer mflux
# can't break the refusal below; tests/test_mflux_drift.py fails when the two stop matching.
def _is_hf_format(path: str | None) -> bool:
    return path is not None and "/" in path and path.count("/") == 1 and not path.startswith(("./", "../", "~/"))


# Copied from mflux's CommandLineParser._option_was_provided (src/mflux/cli/parser/parsers.py), the argv scan its
# commands use to tell whether an option was typed. A copy, so a rename of that private method in a newer mflux can't
# break the Z-Image command's pre-load warning; tests/test_mflux_drift.py fails when the two stop matching.
def _option_was_provided(*option_names: str) -> bool:
    argv = sys.argv[1:]
    for token in argv:
        if token in option_names:
            return True
        for option in option_names:
            if option.startswith("--") and token.startswith(f"{option}="):
                return True
            if (
                option.startswith("-")
                and not option.startswith("--")
                and len(option) == 2
                and token.startswith(option)
                and token != option
            ):
                return True
    return False


# mflux 0.24.0's model-path rules (PathResolution.RULES), as (priority, check). The refusal below copies what they
# decide, so it applies only while mflux runs these rules; any other set leaves every name to mflux's load().
REVIEWED_PATH_RULES = frozenset(
    {(0, "is_none"), (1, "exists_locally"), (2, "is_hf_cached"), (3, "is_hf_format"), (4, "always")}
)


def _path_rules_are_reviewed() -> bool:
    """False for any rule set but the reviewed one, and for rules this code can't read (a reshaped Rule in a newer
    mflux): either way the name is left to mflux."""
    try:
        return {(rule.priority, rule.check) for rule in PathResolution.RULES} == REVIEWED_PATH_RULES
    except (AttributeError, TypeError):
        return False


def _refuse_unknown_model(parser: CommandLineParser, adapter: Adapter, args: Namespace) -> None:
    """mflux's validate gives any --model outside its registry the default entry; its model-path rules then raise
    FileNotFoundError inside load() on a name that is no path that exists and not org/model in shape. Refuse those
    names here, while mflux's rules are the reviewed ones. An org/model name is left to load(): only a download can
    tell whether that repo exists. An empty --model is no path at all: mflux's loader runs the command's default model
    for it."""
    model_path = getattr(args, "model_path", None)
    if not model_path or _is_hf_format(model_path) or os.path.exists(os.path.expanduser(model_path)):
        return
    if not _path_rules_are_reviewed():
        return
    names = ", ".join(supported_model_names(adapter)) or "none"
    # mflux's own tip for an init image a -C file restored (CommandLineParser.parse_args), for the model it restored.
    sidecar = getattr(args, "config_from_metadata", None)
    tip = ""
    if sidecar and not _option_was_provided("--model", "-m"):
        tip = f"\nTip: that path came from {sidecar}. Pass --model PATH to use your own copy."
    parser.error(
        f"Model not found: {model_path!r}. Supported on this command: {names}. "
        f"For other weights, give a local path that exists or a Hugging Face repo id (org/model).{tip}"
    )


def _refuse_what_teacache_cannot_run(
    parser: CommandLineParser, adapter: Adapter, args: Namespace, model_config: ModelConfig
) -> VariantInfo:
    variant = match_variant(model_config, adapter.pipeline)
    if variant is None:
        shown = args.model or model_config.model_name
        names = ", ".join(supported_model_names(adapter)) or "none"
        parser.error(
            f"TeaCache doesn't support {shown} yet. Supported on this command: {names}. "
            f"Run {adapter.plain_command} to generate without TeaCache."
        )
    if variant.default_thresh is None:
        parser.error(
            f"{variant.display_name} is a distilled few-step model: TeaCache has no steps to skip there. "
            f"Run {adapter.plain_command} instead."
        )
    if getattr(args, "step_cache_ratio", None) is not None:
        parser.error(
            "--step-cache-ratio (given on the command line or restored by --config-from-conf) also skips steps, "
            "and mflux's step cache and TeaCache can't run together. "
            "Remove --step-cache-ratio (or --teacache-ratio) from the command line, or the step_cache_ratio key "
            f"from the -C file; or run {adapter.plain_command} to use mflux's step cache."
        )
    active_steps = (adapter.active_steps or default_active_steps)(args, model_config)
    try:
        check_step_window(active_num_steps=active_steps, nominal_num_inference_steps=args.steps)
    # TeaCacheValueError: a negative count (--steps -1, or --steps 0 with --image); mflux's --steps has no range check.
    # The plugin prints its own sentence: the library's text names its internal parameters.
    except (InvalidStepWindowError, TeaCacheValueError) as exc:
        if isinstance(exc, TeaCacheValueError):
            has = f"no valid step count (--steps {args.steps})"
        elif active_steps == args.steps:
            has = f"{active_steps}"
        else:  # an --image strength (or the adapter's own schedule) left fewer steps than --steps
            has = f"{active_steps} of its {args.steps} --steps"
        parser.error(
            "TeaCache needs at least 3 denoising steps: it always computes the first and the last one. "
            f"This run has {has}. Use more --steps (or a lower --image strength), "
            f"or run {adapter.plain_command} without TeaCache."
        )
    return variant


def _warn_unverified_checkpoint(
    adapter: Adapter, args: Namespace, model_config: ModelConfig, variant: VariantInfo
) -> None:
    """One warning, never a refusal, when the coefficients were not fitted on the checkpoint this run loads."""
    model_path = getattr(args, "model_path", None)
    if not variant.calibrated:
        shown = model_path or args.model or model_config.model_name
        warnings.warn(
            TeaCacheUncalibratedCheckpointWarning(
                f"{shown} is not the checkpoint {variant.display_name}'s TeaCache settings were tuned on. "
                "Skip counts and image quality on it are unmeasured. TeaCache still runs."
            ),
            stacklevel=3,
        )
        return
    # Hub repo ids are case-insensitive, and so is mlx-teacache's own comparison.
    known = {model_config.model_name.casefold(), *(repo.casefold() for repo in adapter.calibrated_copies)}
    if not model_path or model_path.casefold() in known:  # "" runs the default model, as no --model does
        return
    warnings.warn(
        TeaCacheUncalibratedCheckpointWarning(
            f"{model_path} may not be {model_config.model_name}, the checkpoint TeaCache's settings were tuned on. "
            "Skip counts and image quality on it are unmeasured. TeaCache still runs."
        ),
        stacklevel=3,
    )


def _warn_unmeasured_compute_precision(args: Namespace) -> None:
    """Reads the parsed value, so a -C sidecar that recorded a compute precision warns too."""
    value = getattr(args, "compute_precision", None)
    if value is not None:
        warnings.warn(COMPUTE_PRECISION_WARNING.format(value=value), UserWarning, stacklevel=3)


def run(adapter: Adapter) -> None:
    parser = adapter.build_parser()
    args = parser.parse_args()
    threshold = _checked_threshold(parser, adapter, args.teacache_threshold)
    if adapter.before_validate is not None:
        adapter.before_validate(args)
    try:
        model_config = adapter.validate(args)
    except ModelConfigError as exc:
        parser.error(str(exc))
    _refuse_unknown_model(parser, adapter, args)
    # TeaCache's refusals come before mflux's option warnings: a refused run prints one message.
    variant = _refuse_what_teacache_cannot_run(parser, adapter, args, model_config)
    if adapter.after_checks is not None:
        adapter.after_checks(args, model_config)
    _warn_unverified_checkpoint(adapter, args, model_config, variant)
    _warn_unmeasured_compute_precision(args)

    model = adapter.load(args)
    memory_saver = None
    try:
        handle = apply_teacache(model, rel_l1_thresh=threshold)
        memory_saver = CallbackManager.register_callbacks(args=args, model=model, latent_creator=adapter.latent_creator)
        for seed in args.seed:
            committed_before = handle.stats.generations
            image = adapter.generate(model, args, seed, PromptUtil.read_prompt(args))
            _record_teacache(image, handle, committed_before)
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    except Exception:  # noqa: BLE001 -- any failure after load: keep the traceback for a report, exit 1
        traceback.print_exc()
        print(
            f"{adapter.command}: error: the run failed (traceback above). "
            f"Run {adapter.plain_command} to generate without TeaCache.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())
