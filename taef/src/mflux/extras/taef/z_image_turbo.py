"""mflux-generate-z-image-turbo-taef: mflux's Z-Image Turbo command with live TAEF previews."""

from argparse import Namespace

from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.vae.vae_decode_options import VAEDecodeOptions
from mflux.models.z_image.cli import z_image_turbo_generate
from mflux.models.z_image.cli.z_image_turbo_generate import CONDITIONAL_OPTIONS, IGNORED_OPTIONS, ZImageTurboCommand
from mflux.models.z_image.variants.z_image import ZImage

from ._core import Adapter, add_taef_arguments, run

__all__ = [
    "ADAPTER",
    "CONDITIONAL_OPTIONS",
    "DESCRIPTION",
    "IGNORED_OPTIONS",
    "PIPELINE",
    "build_parser",
    "main",
    "warn_before_load",
]

# The class ZImageTurboCommand.load() returns.
PIPELINE = ZImage
DESCRIPTION = "Generate an image using Z-Image Turbo with live TAEF previews."


def build_parser() -> CommandLineParser:
    parser = z_image_turbo_generate.build_parser()
    parser.description = DESCRIPTION
    add_taef_arguments(parser)
    return parser


# The two statements below are copied from mflux's z_image_turbo_generate.main(), which runs them right after
# parsing. Here they run after the plugin's refusals, so a refused run prints only the refusal.
# tests/test_mflux_drift.py fails when they stop matching.
def warn_before_load(args: Namespace, model_config: ModelConfig) -> None:
    VAEDecodeOptions.warn_encode_only_tiling()
    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)


ADAPTER = Adapter(
    command="mflux-generate-z-image-turbo-taef",
    plain_command="mflux-generate-z-image-turbo",
    build_parser=build_parser,
    validate=ZImageTurboCommand.validate,
    load=ZImageTurboCommand.load,
    generate=ZImageTurboCommand.generate,
    latent_creator=ZImageTurboCommand.latent_creator,
    after_checks=warn_before_load,
)


def main() -> None:
    run(ADAPTER)
