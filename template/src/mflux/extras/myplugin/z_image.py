"""mflux-generate-z-image-myplugin: mflux's Z-Image command with the myplugin plugin attached."""

import warnings
from argparse import Namespace
from typing import Any

from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.z_image.cli import z_image_generate
from mflux.models.z_image.cli.z_image_generate import CONDITIONAL_OPTIONS, ZImageCommand

from ._core import Adapter, run

__all__ = [
    "ADAPTER",
    "CONDITIONAL_OPTIONS",
    "DESCRIPTION",
    "OPTION_FLAG",
    "apply",
    "build_parser",
    "main",
    "refuse",
    "warn_ineffective_options",
]

DESCRIPTION = "Generate an image using Z-Image with the myplugin plugin."
OPTION_FLAG = "--myplugin-option"


def build_parser() -> CommandLineParser:
    parser = z_image_generate.build_parser()
    parser.description = DESCRIPTION
    group = parser.add_argument_group("myplugin")
    group.add_argument(
        OPTION_FLAG,
        dest="myplugin_option",
        type=float,
        default=None,
        metavar="VALUE",
        help="Example option: replace it with your plugin's own. Greater than 0. Default: not set.",
    )
    return parser


def refuse(parser: CommandLineParser, args: Namespace, model_config: ModelConfig) -> None:
    """Refuse, before any weights load, what the plugin can't run. parser.error exits with code 2."""
    value = args.myplugin_option
    if value is not None and not value > 0:  # NaN fails the comparison and is refused
        parser.error(f"{OPTION_FLAG} must be greater than 0, got {value}. Run {ADAPTER.plain_command} instead.")


# The function below is copied statement for statement from mflux's z_image_generate.main(),
# which runs it before load(). tests/test_mflux_drift.py fails when it stops matching.
def warn_ineffective_options(args: Namespace, model_config: ModelConfig) -> None:
    # Warn on the EFFECTIVE behavior, not the flag value: --model may resolve to a
    # guidance-distilled variant (guidance forced to 0.0 regardless of the flag), and on
    # CFG-capable models the negative prompt is only encoded at guidance > 1.0, where an
    # omitted --guidance defaults to 0.0.
    if not model_config.supports_guidance:
        CommandLineParser.warn_ignored_options(
            {
                "--guidance": CONDITIONAL_OPTIONS["--guidance"]["reason"],
                "--negative-prompt": CONDITIONAL_OPTIONS["--negative-prompt"]["reason"],
            }
        )
    elif CommandLineParser._option_was_provided("--negative-prompt") and (
        args.guidance is None or args.guidance <= 1.0
    ):
        warnings.warn(
            f"--negative-prompt has no effect: {CONDITIONAL_OPTIONS['--negative-prompt']['reason']}"
            " Pass --guidance above 1.0 to enable it.",
            stacklevel=1,
        )


def apply(model: Any, args: Namespace) -> Any:
    """Attach your library to the loaded model here, once per run, before the first image, and return the plugin's
    handle (before_generate and after_generate receive it). The template attaches nothing and returns None, so the
    command behaves exactly like mflux-generate-z-image. To change the prediction step, replace model._predict with a
    factory (transformer) -> predict that builds the step from the transformer mflux passes for each image, never
    from model.transformer. Read "Compile and memory" in AGENTS.md before you attach anything: mflux may compile the
    prediction step, and the handle must not keep the transformer alive."""
    return None


ADAPTER = Adapter(
    command="mflux-generate-z-image-myplugin",
    plain_command="mflux-generate-z-image",
    build_parser=build_parser,
    validate=ZImageCommand.validate,
    load=ZImageCommand.load,
    generate=ZImageCommand.generate,
    latent_creator=ZImageCommand.latent_creator,
    apply=apply,
    refuse=refuse,
    after_checks=warn_ineffective_options,
)


def main() -> None:
    run(ADAPTER)
