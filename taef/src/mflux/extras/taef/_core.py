"""The command flow every mflux-taef command shares.

Each model family has a small adapter module (z_image.py, z_image_turbo.py) that names mflux's own command steps.
run() parses with mflux's parser plus the "TAEF preview" options, refuses what the plugin can't run before any weights
load, then loads, generates and saves the way the plain mflux command does."""

import os
import re
import string
import sys
import traceback
from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.utils.exceptions import ModelConfigError, PromptFileReadError, StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.prompt_util import PromptUtil
from mlx_taef.errors import UnsupportedMfluxModelError
from mlx_taef.integrations.mflux import LivePreviewCallback
from mlx_taef.kernels import resolve_kernel_from_model_config

PREVIEW_FLAG = "--taef-preview"
EVERY_FLAG = "--taef-every"
FRAMES_FLAG = "--taef-frames"
PREVIEW_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
DEFAULT_EVERY = 1


@dataclass(frozen=True)
class Adapter:
    """One mflux command as run() needs it: its names and mflux's own command steps, plus one optional hook.
    - after_checks(args, model_config): after the plugin's refusals, before load; mflux's own pre-load warnings, so a
      refused run prints one message."""

    command: str
    plain_command: str
    build_parser: Callable[[], CommandLineParser]
    validate: Callable[[Namespace], ModelConfig]
    load: Callable[[Namespace], Any]
    generate: Callable[[Any, Namespace, int, str], GeneratedImage]
    latent_creator: type
    after_checks: Callable[[Namespace, ModelConfig], None] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class PreviewOptions:
    """The checked preview settings: decode every `every` steps; keep each step as its own file when `frames`."""

    every: int
    frames: bool


def add_taef_arguments(parser: CommandLineParser) -> None:
    group = parser.add_argument_group("TAEF preview")
    group.add_argument(
        PREVIEW_FLAG,
        dest="taef_preview",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Where to write the live preview: a .png, .jpg, .jpeg or .webp file. It may contain {seed}. "
            "Default: next to the output image, named <output name>.preview.png. A path without {seed} is one file "
            "that each seed overwrites in turn; with --taef-frames and several seeds, _seed_<seed> is added to its "
            "name instead."
        ),
    )
    group.add_argument(
        EVERY_FLAG,
        dest="taef_every",
        type=int,
        default=None,
        metavar="N",
        help="Update the preview every N denoising steps, 1 or more. Default: every step.",
    )
    group.add_argument(
        FRAMES_FLAG,
        dest="taef_frames",
        action="store_true",
        help=(
            "Keep every step's preview as its own file, <preview name>_stepNN.<suffix>, instead of overwriting one "
            "file. Writes every step, so it can't be combined with --taef-every."
        ),
    )


def _format_fields(template: str) -> list[str]:
    """The field names of the brace fields in `template`; ValueError when a brace is unmatched."""
    return [field for _, field, _, _ in string.Formatter().parse(template) if field is not None]


# The one format spec a {seed} field may carry: a width of one to three digits in all, a leading zero counted (padded
# with spaces, or with zeros when it starts with 0), with an optional d ({seed:04d}). Anything else int takes could
# build a huge path ({seed:>2000000000}) or a folder separator ({seed:c} for seed 47). A three-digit width can still
# make a name over the 255 bytes a file name may have ({seed:999}); checked_preview_options refuses that name.
SEED_SPEC = re.compile(r"[0-9]{1,3}d?")
# The longest file name, in bytes, that APFS and the common Linux file systems take.
NAME_MAX_BYTES = 255


def _only_plain_seed_fields(template: str) -> bool:
    """True when every brace field in `template` is {seed}, with no conversion and at most a SEED_SPEC width."""
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError:
        return False
    return all(
        field == "seed" and conversion is None and (not spec or SEED_SPEC.fullmatch(spec) is not None)
        for _, field, spec, conversion in parsed
        if field is not None
    )


def preview_path(args: Namespace, seed: int) -> Path:
    """Where this seed's preview goes. The default sits next to the output and inherits the _seed_ suffix mflux's
    parser puts on --output for several seeds. An explicit path gets _seed_{seed} added to its stem only in frames
    mode with several seeds and no {seed} of its own; otherwise it is the same file for every seed."""
    if args.taef_preview is None:
        output = Path(args.output.format(seed=seed))
        return output.with_name(output.stem + ".preview.png")
    template = args.taef_preview
    if args.taef_frames and len(args.seed) > 1 and "seed" not in _format_fields(template):
        explicit = Path(template)
        template = str(explicit.with_stem(explicit.stem + "_seed_{seed}"))
    return Path(template.format(seed=seed))


def same_file_key(path: Path) -> str:
    """One key per file on a default APFS volume, which is case-insensitive: Path.resolve() keeps the typed case, so
    A.PNG and a.png would compare as different files."""
    return str(path.resolve()).casefold()


def _frame_name_pattern(preview: Path) -> re.Pattern[str]:
    """The case-folded names frames mode writes for `preview`: <preview stem>_step<digits><suffix>."""
    stem, suffix = preview.stem.casefold(), preview.suffix.casefold()
    return re.compile(f"{re.escape(stem)}_step[0-9]+{re.escape(suffix)}")


def output_is_a_preview_file(preview: Path, output: Path, *, frames: bool) -> bool:
    """Whether a preview this seed writes is the file `output` names. One file is the preview path itself; in frames
    mode the files are <preview stem>_step<digits><suffix> in the preview's folder, and the preview path is never
    written."""
    if not frames:
        return same_file_key(preview) == same_file_key(output)
    if same_file_key(preview.parent) != same_file_key(output.parent):
        return False
    return _frame_name_pattern(preview).fullmatch(output.name.casefold()) is not None


def written_name(preview: Path, *, frames: bool, steps: int) -> str:
    """The longest file name this preview writes: the preview's own name, or in frames mode the name of the last
    step's frame (mlx-taef numbers frames from _step00, two digits at least)."""
    if not frames:
        return preview.name
    return f"{preview.stem}_step{max(steps - 1, 0):02d}{preview.suffix}"


def frame_that_is_not_a_file(preview: Path) -> Path | None:
    """An existing directory or symbolic link at one of the frame names next to `preview`, or None. mlx-taef's save
    follows a link and fails on a directory."""
    pattern = _frame_name_pattern(preview)
    try:
        with os.scandir(preview.parent) as entries:
            names = sorted(
                entry.name
                for entry in entries
                if pattern.fullmatch(entry.name.casefold()) and (entry.is_symlink() or entry.is_dir())
            )
    except OSError:  # no such folder yet (the frames make it), or one this user can't list: no frame exists there
        return None
    return preview.parent / names[0] if names else None


def checked_preview_options(parser: CommandLineParser, adapter: Adapter, args: Namespace) -> PreviewOptions:
    """The preview options, or exit 2 (parser.error) on one this command can't run. Builds nothing."""
    hint = f"Run {adapter.plain_command} to generate without previews."
    every = args.taef_every
    if every is not None and every < 1:
        parser.error(f"{EVERY_FLAG} must be 1 or more, got {every}. {hint}")
    if every is not None and args.taef_frames:
        parser.error(f"{EVERY_FLAG} and {FRAMES_FLAG} can't be used together: {FRAMES_FLAG} writes every step.")
    explicit = args.taef_preview
    if explicit is not None:
        if not _only_plain_seed_fields(explicit):
            parser.error(f"{PREVIEW_FLAG} can contain {{seed}} and no other {{...}} field, got {explicit}. {hint}")
        # The suffix the stem change in preview_path keeps; read before it, which fails on a path without a name.
        if Path(explicit.format(seed=args.seed[0])).suffix.lower() not in PREVIEW_SUFFIXES:
            parser.error(f"{PREVIEW_FLAG} must end in .png, .jpg, .jpeg or .webp, got {explicit}. {hint}")
    # Every seed's output: with {seed} in --taef-preview, one seed's preview can be another seed's output.
    outputs = [Path(args.output.format(seed=seed)) for seed in args.seed]
    for seed in args.seed:
        preview = preview_path(args, seed)
        name = written_name(preview, frames=args.taef_frames, steps=args.steps)
        size = len(os.fsencode(name))
        if size > NAME_MAX_BYTES:
            # Saved as it is, the name fails with ENAMETOOLONG once the model has loaded, and the preview turns off.
            parser.error(
                f"{PREVIEW_FLAG} makes a file name of {size} bytes, over the {NAME_MAX_BYTES} a file name may have: "
                f"{name}. Give the preview a shorter name."
            )
        # os.path rather than Path: its checks answer False on any OSError, such as a name too long for the volume.
        if os.path.isdir(preview) or os.path.islink(preview):
            parser.error(f"{PREVIEW_FLAG} must name a file, not a directory or a symbolic link, got {preview}.")
        frame = frame_that_is_not_a_file(preview) if args.taef_frames else None
        if frame is not None:
            parser.error(f"{PREVIEW_FLAG} frames must be files, not a directory or a symbolic link, got {frame}.")
        if any(output_is_a_preview_file(preview, output, frames=args.taef_frames) for output in outputs):
            shown = preview.with_name(f"{preview.stem}_stepNN{preview.suffix}") if args.taef_frames else preview
            parser.error(f"{PREVIEW_FLAG} names the same file as --output ({shown}). Give the preview its own name.")
    return PreviewOptions(every=DEFAULT_EVERY if every is None else every, frames=args.taef_frames)


def preview_variant(parser: CommandLineParser, adapter: Adapter, args: Namespace, model_config: ModelConfig) -> str:
    """The mlx-taef decoder for this model, or exit 2 when mlx-taef has none. It loads no weights."""
    try:
        return resolve_kernel_from_model_config(model_config).name
    except UnsupportedMfluxModelError:
        shown = args.model or model_config.model_name
        parser.error(
            f"TAEF has no preview decoder for {shown}. Run {adapter.plain_command} to generate without previews."
        )


# Copied from mflux's PathResolution._is_hf_format (src/mflux/models/common/resolution/path_resolution.py), the check
# its model-path rules use for a Hugging Face repo id. A copy, so a rename of that private method in a newer mflux
# can't break the refusal below; tests/test_mflux_drift.py fails when the two stop matching.
def _is_hf_format(path: str | None) -> bool:
    return path is not None and "/" in path and path.count("/") == 1 and not path.startswith(("./", "../", "~/"))


# Copied from mflux's CommandLineParser._option_was_provided (src/mflux/cli/parser/parsers.py), the argv scan its
# commands use to tell whether an option was typed. A copy, so a rename of that private method in a newer mflux can't
# break the base command's pre-load warning; tests/test_mflux_drift.py fails when the two stop matching.
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
    """False for any rule set but the checked one, and for rules this code can't read (a reshaped Rule in a newer
    mflux): either way the name is left to mflux."""
    try:
        return {(rule.priority, rule.check) for rule in PathResolution.RULES} == REVIEWED_PATH_RULES
    except (AttributeError, TypeError):
        return False


def _refuse_unknown_model(parser: CommandLineParser, args: Namespace) -> None:
    """mflux's validate gives any --model outside its registry the default entry; its model-path rules then raise
    FileNotFoundError inside load() on a name that is no path that exists and not org/model in shape. Refuse those
    names here, while mflux's rules are the checked ones. An org/model name is left to load(): only a download can
    tell whether that repo exists. An empty --model is no path at all: mflux's loader runs the command's default model
    for it."""
    model_path = getattr(args, "model_path", None)
    if not model_path or _is_hf_format(model_path) or os.path.exists(os.path.expanduser(model_path)):
        return
    if not _path_rules_are_reviewed():
        return
    # mflux's own tip for an init image a -C file restored (CommandLineParser.parse_args), for the model it restored.
    sidecar = getattr(args, "config_from_metadata", None)
    tip = ""
    if sidecar and not _option_was_provided("--model", "-m"):
        tip = f"\nTip: that path came from {sidecar}. Pass --model PATH to use your own copy."
    parser.error(
        f"Model not found: {model_path!r}. "
        f"Give a model name this command knows, a local path that exists or a Hugging Face repo id (org/model).{tip}"
    )


def build_preview(*, variant: str, options: PreviewOptions, first_path: Path) -> LivePreviewCallback:
    """mlx-taef's live preview callback; building it prepares the decoder (a download on first use). Built without
    the model: the variant is already resolved, and the model loads only after this has succeeded."""
    return LivePreviewCallback(
        # A kernel name mlx-taef itself resolved; the constructor rejects any name it has no decoder for.
        variant=cast("Any", variant),
        every=options.every,
        save_to=first_path,
        numbered_frames=options.frames,
        on_error="disable",
    )


def start_line(path: Path, options: PreviewOptions, variant: str) -> str:
    """The line that tells the user where this seed's preview goes, how often it updates and which decoder runs."""
    decoder = f"({variant} decoder)"
    if options.frames:
        frames = path.with_name(f"{path.stem}_stepNN{path.suffix}")
        return f"TAEF preview frames: {frames}, every step {decoder}"
    cadence = "every step" if options.every == 1 else f"every {options.every} steps"
    return f"TAEF preview: {path}, {cadence} {decoder}"


def run(adapter: Adapter) -> None:
    parser = adapter.build_parser()
    args = parser.parse_args()
    options = checked_preview_options(parser, adapter, args)
    try:
        model_config = adapter.validate(args)
    except ModelConfigError as exc:
        parser.error(str(exc))
    _refuse_unknown_model(parser, args)
    variant = preview_variant(parser, adapter, args, model_config)
    # The plugin's refusals come before mflux's option warnings: a refused run prints one message.
    if adapter.after_checks is not None:
        adapter.after_checks(args, model_config)
    # The decoder is prepared before the model loads, so a missing one costs seconds, not a model load.
    try:
        preview = build_preview(variant=variant, options=options, first_path=preview_path(args, args.seed[0]))
    except Exception as exc:  # noqa: BLE001 -- any failure to prepare the decoder: one line, exit 1, before load
        print(
            f"{adapter.command}: error: could not prepare the TAEF preview decoder: {exc}. "
            f"Run {adapter.plain_command} to generate without previews.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    model = adapter.load(args)
    memory_saver = None
    try:
        # Registered first, so the preview runs ahead of mflux's own callbacks in every hook.
        model.callbacks.register(preview)
        memory_saver = CallbackManager.register_callbacks(args=args, model=model, latent_creator=adapter.latent_creator)
        for seed in args.seed:
            # Always a Path: the callback derives its numbered-frame names from save_to.
            preview.save_to = preview_path(args, seed)
            print(start_line(preview.save_to, options, variant))
            image = adapter.generate(model, args, seed, PromptUtil.read_prompt(args))
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    except Exception:  # noqa: BLE001 -- any failure after load: keep the traceback for a report, exit 1
        traceback.print_exc()
        print(
            f"{adapter.command}: error: the run failed (traceback above). "
            f"Run {adapter.plain_command} to generate without previews.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())
