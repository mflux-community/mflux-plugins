"""mflux-generate-qwen-teacache: mflux's Qwen-Image command with TeaCache step skipping."""

from argparse import Namespace

from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.qwen.cli import qwen_image_generate
from mflux.models.qwen.cli.qwen_image_generate import QwenImageCommand
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.utils.exceptions import ModelConfigError

from ._core import Adapter, add_teacache_arguments, run

__all__ = ["ADAPTER", "DESCRIPTION", "ORIGINAL_REPO", "PIPELINE", "build_parser", "main", "validate"]

# The class QwenImageCommand.load() returns; mlx-teacache's match_variant is asked about it.
PIPELINE = QwenImage
DESCRIPTION = "Generate an image using Qwen Image model with TeaCache step skipping."
# The original Qwen-Image checkpoint. mflux's Qwen-Image command loads any repo it is given on the Qwen-Image-2512
# config, and mlx-teacache cannot tell these weights from 2512 there.
ORIGINAL_REPO = "Qwen/Qwen-Image"
# The org's copies of the original model at other precisions: each model card declares base_model: Qwen/Qwen-Image
# (checked on the Hub 2026-10-23).
ORIGINAL_COPIES = (
    "mflux-community/qwen-image-mflux-q3",
    "mflux-community/qwen-image-mflux-q4",
    "mflux-community/qwen-image-mflux-q5",
    "mflux-community/qwen-image-mflux-q6",
    "mflux-community/qwen-image-mflux-q8",
    "mflux-community/qwen-image-mflux-bf16",
)


def build_parser() -> CommandLineParser:
    parser = qwen_image_generate.build_parser()
    parser.description = DESCRIPTION
    add_teacache_arguments(parser)
    return parser


def validate(args: Namespace) -> ModelConfig:
    """mflux's QwenImageCommand.validate, after refusing the original Qwen-Image weights (Hub repo ids, compared
    without regard to case). A local folder can't be identified; it runs with the checkpoint warning."""
    model_path = getattr(args, "model_path", None)
    original = {repo.casefold() for repo in (ORIGINAL_REPO, *ORIGINAL_COPIES)}
    if model_path is not None and model_path.casefold() in original:
        raise ModelConfigError(
            f"{model_path} holds the original Qwen-Image. This command runs Qwen-Image-2512, and its TeaCache "
            "settings were tuned for that model. To run the original Qwen-Image with TeaCache, use mlx-teacache from "
            "Python."
        )
    return QwenImageCommand.validate(args)


# Qwen-Image needs no optional hook: mflux's main() has no pre-load step, and active_steps keeps the Config-based
# default. mlx-teacache cannot tell custom Qwen weights from Qwen-Image-2512, so this package's own model-path
# warning applies, except on the known copies of the calibrated checkpoint.
ADAPTER = Adapter(
    command="mflux-generate-qwen-teacache",
    plain_command="mflux-generate-qwen",
    pipeline=PIPELINE,
    family_keys=(qwen_image_generate.DEFAULT_MODEL,),
    build_parser=build_parser,
    validate=validate,
    load=QwenImageCommand.load,
    generate=QwenImageCommand.generate,
    latent_creator=QwenImageCommand.latent_creator,
    # The org's copies of Qwen/Qwen-Image-2512 at other precisions: each model card declares
    # base_model: Qwen/Qwen-Image-2512 (checked on the Hub 2026-10-23). TeaCache's settings were measured on one
    # build only.
    calibrated_copies=(
        "mflux-community/qwen-image-2512-mflux-q3",
        "mflux-community/qwen-image-2512-mflux-q4",
        "mflux-community/qwen-image-2512-mflux-q5",
        "mflux-community/qwen-image-2512-mflux-q6",
        "mflux-community/qwen-image-2512-mflux-q8",
        "mflux-community/qwen-image-2512-mflux-bf16",
    ),
)


def main() -> None:
    run(ADAPTER)
