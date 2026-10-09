"""mflux-generate-z-image-teacache: mflux's Z-Image command with TeaCache step skipping."""

import warnings
from argparse import Namespace

from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.z_image.cli import z_image_generate
from mflux.models.z_image.cli.z_image_generate import CONDITIONAL_OPTIONS, ZImageCommand
from mflux.models.z_image.variants.z_image import ZImage

from ._core import Adapter, add_teacache_arguments, run

__all__ = [
    "ADAPTER",
    "CONDITIONAL_OPTIONS",
    "DESCRIPTION",
    "PIPELINE",
    "build_parser",
    "main",
    "warn_ineffective_options",
]

# The class ZImageCommand.load() returns; mlx-teacache's match_variant is asked about it.
PIPELINE = ZImage
DESCRIPTION = "Generate an image using Z-Image with TeaCache step skipping."


def build_parser() -> CommandLineParser:
    parser = z_image_generate.build_parser()
    parser.description = DESCRIPTION
    add_teacache_arguments(parser)
    return parser


# The function below is copied statement for statement from mflux's z_image_generate.main(),
# which runs it before load(). tests/test_mflux_drift.py fails when it stops matching. It goes
# away once mflux's Z-Image command has a Command.prepare(args) step the plugin can call.
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


# Z-Image needs one optional hook (after_checks); active_steps keeps the Config-based default, and
# mlx-teacache has no checkpoint check for Z-Image, so this package's own model-path warning applies,
# except on the known copies of the calibrated checkpoint.
ADAPTER = Adapter(
    command="mflux-generate-z-image-teacache",
    plain_command="mflux-generate-z-image",
    pipeline=PIPELINE,
    family_keys=(z_image_generate.DEFAULT_MODEL, *z_image_generate.FAMILY_MODELS),
    build_parser=build_parser,
    validate=ZImageCommand.validate,
    load=ZImageCommand.load,
    generate=ZImageCommand.generate,
    latent_creator=ZImageCommand.latent_creator,
    after_checks=warn_ineffective_options,
    # The org's copies of Tongyi-MAI/Z-Image at other precisions (q3 to q8 quantized, bf16 not): each model card
    # declares base_model: Tongyi-MAI/Z-Image (checked on the Hub 2026-10-09). TeaCache's settings were measured on
    # one 8-bit build only. The z-image-turbo-* copies hold Turbo, a different model, and are not here.
    calibrated_copies=(
        "mflux-community/z-image-base-mflux-q3",
        "mflux-community/z-image-base-mflux-q4",
        "mflux-community/z-image-base-mflux-q5",
        "mflux-community/z-image-base-mflux-q6",
        "mflux-community/z-image-base-mflux-q8",
        "mflux-community/z-image-base-mflux-bf16",
    ),
)


def main() -> None:
    run(ADAPTER)
