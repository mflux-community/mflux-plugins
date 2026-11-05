"""The command flow every mflux-taef command shares.

Each model family has a small adapter module (z_image.py, z_image_turbo.py) that names mflux's own command steps.
run() parses with mflux's parser plus the "TAEF preview" options, refuses what the plugin can't run before any weights
load, then loads, generates and saves the way the plain mflux command does."""

import os
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
        bad_field = f"{PREVIEW_FLAG} can contain {{seed}} and no other {{...}} field, got {explicit}. {hint}"
        try:
            fields = _format_fields(explicit)
        except ValueError:
            fields = [""]
        if any(field != "seed" for field in fields):
            parser.error(bad_field)
        try:
            # A {seed} field can still carry a format spec or conversion a seed can't take ({seed:x.y}, {seed!z}), or
            # a field nested in its format spec ({seed:{w}}).
            formatted = explicit.format(seed=args.seed[0])
        except (ValueError, LookupError, AttributeError, TypeError):
            parser.error(bad_field)
        # The suffix the stem change in preview_path keeps; read before it, which fails on a path without a name.
        if Path(formatted).suffix.lower() not in PREVIEW_SUFFIXES:
            parser.error(f"{PREVIEW_FLAG} must end in .png, .jpg, .jpeg or .webp, got {explicit}. {hint}")
    for seed in args.seed:
        preview = preview_path(args, seed)
        if same_file_key(preview) == same_file_key(Path(args.output.format(seed=seed))):
            parser.error(f"{PREVIEW_FLAG} names the same file as --output ({preview}). Give the preview its own name.")
    return PreviewOptions(every=DEFAULT_EVERY if every is None else every, frames=args.taef_frames)


def preview_variant(parser: CommandLineParser, adapter: Adapter, args: Namespace, model_config: ModelConfig) -> str:
    """The mlx-taef decoder for this model, or exit 2 when mlx-taef has none. Weight-free."""
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


def _refuse_unknown_model(parser: CommandLineParser, args: Namespace) -> None:
    """mflux's validate gives any --model outside its registry the default entry; its model-path rules then raise
    FileNotFoundError inside load() on a name that is no path that exists and not org/model in shape. Refuse those
    names here. An org/model name is left to load(): only a download can tell whether that repo exists. An empty
    --model is no path at all: mflux's loader runs the command's default model for it."""
    model_path = getattr(args, "model_path", None)
    if not model_path or _is_hf_format(model_path) or os.path.exists(os.path.expanduser(model_path)):
        return
    parser.error(
        f"Model not found: {model_path!r}. "
        "Give a model name this command knows, a local path that exists or a Hugging Face repo id (org/model)."
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
        return f"TAEF preview frames: {path.stem}_stepNN{path.suffix} next to {path}, every step {decoder}"
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
