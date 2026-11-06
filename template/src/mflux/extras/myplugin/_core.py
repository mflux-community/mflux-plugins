"""The command flow a plugin command shares with the mflux command it extends.

Each model family has a small adapter module (z_image.py) that names mflux's own command steps. run()
parses with mflux's parser, refuses what the plugin can't run before any weights load, loads with mflux,
attaches the plugin to the loaded model, then generates and saves the way the plain mflux command does."""

import os
import sys
import traceback
from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.utils.exceptions import ModelConfigError, PromptFileReadError, StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.prompt_util import PromptUtil


@dataclass(frozen=True)
class Adapter:
    """One mflux command as run() needs it.

    The required fields are the command names, mflux's command steps, and apply:
    - apply(model, args) -> handle: once per run, after load and before mflux registers its callbacks;
      attach the plugin's library to the loaded model. Return what the later hooks need (the library's
      handle), or None.
    The hooks are optional:
    - refuse(parser, args, model_config): before load; call parser.error for anything the plugin can't run.
    - after_checks(args, model_config): before load, after refuse; mflux's own pre-load warnings.
    - before_generate(model, args, handle, seed) -> state: before each image; return per-seed state
      (for example a counter read before the run), or None.
    - after_generate(image, args, handle, state): after each image, before it is saved; add the plugin's
      record to a new generation_parameters dict (never mutate the one the image holds: images can share it)."""

    command: str
    plain_command: str
    build_parser: Callable[[], CommandLineParser]
    validate: Callable[[Namespace], ModelConfig]
    load: Callable[[Namespace], Any]
    generate: Callable[[Any, Namespace, int, str], GeneratedImage]
    latent_creator: type
    apply: Callable[[Any, Namespace], Any]
    refuse: Callable[[CommandLineParser, Namespace, ModelConfig], None] | None = None
    after_checks: Callable[[Namespace, ModelConfig], None] | None = None
    before_generate: Callable[[Any, Namespace, Any, int], Any] | None = None
    after_generate: Callable[[GeneratedImage, Namespace, Any, Any], None] | None = None


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


def _refuse_unknown_model(parser: CommandLineParser, args: Namespace) -> None:
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
    parser.error(
        f"Model not found: {model_path!r}. "
        "Give a model name this command knows, a local path that exists or a Hugging Face repo id (org/model)."
    )


def run(adapter: Adapter) -> None:
    parser = adapter.build_parser()
    args = parser.parse_args()
    try:
        model_config = adapter.validate(args)
    except ModelConfigError as exc:
        parser.error(str(exc))
    _refuse_unknown_model(parser, args)
    # The plugin's refusals come before mflux's option warnings: a refused run prints one message.
    if adapter.refuse is not None:
        adapter.refuse(parser, args, model_config)
    if adapter.after_checks is not None:
        adapter.after_checks(args, model_config)

    model = adapter.load(args)
    memory_saver = None
    try:
        # Before mflux's callbacks: a callback the library registers runs ahead of MemorySaver's after-loop step.
        handle = adapter.apply(model, args)
        memory_saver = CallbackManager.register_callbacks(args=args, model=model, latent_creator=adapter.latent_creator)
        for seed in args.seed:
            state = None if adapter.before_generate is None else adapter.before_generate(model, args, handle, seed)
            image = adapter.generate(model, args, seed, PromptUtil.read_prompt(args))
            if adapter.after_generate is not None:
                adapter.after_generate(image, args, handle, state)
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    except Exception:  # noqa: BLE001 -- any failure after load: keep the traceback for a report, exit 1
        traceback.print_exc()
        print(
            f"{adapter.command}: error: the run failed (traceback above). "
            f"Run {adapter.plain_command} to generate without the plugin.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())
