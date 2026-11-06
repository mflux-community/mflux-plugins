"""mflux-generate-z-image-taef: mflux's Z-Image command with live TAEF previews."""

import warnings
from argparse import Namespace

from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.vae.vae_decode_options import VAEDecodeOptions
from mflux.models.z_image.cli import z_image_generate
from mflux.models.z_image.cli.z_image_generate import CONDITIONAL_OPTIONS, ZImageCommand

from ._core import Adapter, _option_was_provided, add_taef_arguments, run

__all__ = [
    "ADAPTER",
    "CONDITIONAL_OPTIONS",
    "DESCRIPTION",
    "build_parser",
    "main",
    "warn_before_load",
    "warn_ineffective_options",
]

DESCRIPTION = "Generate an image using Z-Image with live TAEF previews."


def build_parser() -> CommandLineParser:
    parser = z_image_generate.build_parser()
    parser.description = DESCRIPTION
    add_taef_arguments(parser)
    return parser


# The function below is copied statement for statement from mflux's z_image_generate.main(),
# which runs it before load(), with one change: it calls _core's copy of the argv scan where mflux
# calls CommandLineParser._option_was_provided. tests/test_mflux_drift.py fails when it stops matching.
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
    elif _option_was_provided("--negative-prompt") and (args.guidance is None or args.guidance <= 1.0):
        warnings.warn(
            f"--negative-prompt has no effect: {CONDITIONAL_OPTIONS['--negative-prompt']['reason']}"
            " Pass --guidance above 1.0 to enable it.",
            stacklevel=1,
        )


def warn_before_load(args: Namespace, model_config: ModelConfig) -> None:
    """mflux's pre-load warnings for Z-Image, in the order its main() gives them: first the warning that --vae-tiling
    no longer changes the decode, then the guidance and negative-prompt warnings. It runs after the plugin's refusals,
    not right after parsing as in mflux's main(), so a refused run prints only the refusal."""
    VAEDecodeOptions.warn_encode_only_tiling()
    warn_ineffective_options(args, model_config)


ADAPTER = Adapter(
    command="mflux-generate-z-image-taef",
    plain_command="mflux-generate-z-image",
    build_parser=build_parser,
    validate=ZImageCommand.validate,
    load=ZImageCommand.load,
    generate=ZImageCommand.generate,
    latent_creator=ZImageCommand.latent_creator,
    after_checks=warn_before_load,
)


def main() -> None:
    run(ADAPTER)
