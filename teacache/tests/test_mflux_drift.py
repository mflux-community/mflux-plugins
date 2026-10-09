"""Pins the parts of mflux's Z-Image command this package calls or copies.

When one fails after an mflux update: read mflux's diff of
src/mflux/models/z_image/cli/z_image_generate.py, bring z_image.py (and _core.run, if the flow
after load() changed) in line with it, then record the digest the failure message prints."""

import inspect

import pytest
from _fingerprint import ast_fingerprint, function_body_dumps, statements_fingerprint_from
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.z_image.cli import z_image_generate
from mflux.models.z_image.cli.z_image_generate import ZImageCommand
from mflux.models.z_image.variants.z_image import ZImage
from mflux.utils.generated_image import GeneratedImage

from mflux.extras.teacache import z_image

# Recorded against mflux v.0.22.0 (PyPI 0.22.0, commit 31c9640); same digest on CPython 3.10, 3.13 and 3.14.
BUILD_PARSER_DIGEST = "664a51c06e0e4b7a"
MAIN_DIGEST = "38cfe3fe3f1142f7"
# ZImage.generate_image, the method tests/_fakes.py's FakeZImage.generate_image reproduces up to the image record.
GENERATE_IMAGE_DIGEST = "a0e64be2aa3d8bc9"

# mflux v.0.22.0: every option string and dest of the Z-Image parser, sorted. build_parser's digest sees only the
# helper calls it makes; an option added inside a shared helper (add_image_generator_arguments and the like) shows up
# only here.
OPTION_STRINGS = (
    "--auto-seeds", "--bake-lora", "--base-model", "--battery-percentage-stop-limit", "--compute-precision",
    "--config-from-conf", "--config-from-metadata", "--float32", "--guidance", "--height", "--help", "--image",
    "--image-path", "--image-strength", "--lora", "--lora-paths", "--lora-scales", "--lora-style", "--low-ram",
    "--make-conf", "--metadata", "--mlx-cache-limit-gb", "--model", "--negative-prompt", "--no-bake-lora",
    "--no-exif", "--no-float32", "--no-metadata", "--no-pid-decode", "--output", "--pid-decode",
    "--pid-degrade-sigma", "--prompt", "--prompt-file", "--quantize", "--scheduler", "--seed", "--steps",
    "--stepwise-image-output-dir", "--vae-tile-size", "--vae-tiling", "--verbose", "--width", "-B", "-C", "-h", "-m",
    "-q", "-v",
)  # fmt: skip
DESTS = (
    "auto_seeds", "bake_lora", "base_model", "battery_percentage_stop_limit", "compute_precision",
    "config_from_metadata", "float32", "guidance", "height", "help", "image", "image_path", "image_strength", "lora",
    "lora_paths", "lora_scales", "lora_style", "low_ram", "metadata", "mlx_cache_limit_gb", "model", "negative_prompt",
    "no_metadata", "output", "pid_decode", "pid_degrade_sigma", "prompt", "prompt_file", "quantize", "scheduler",
    "seed", "steps", "stepwise_image_output_dir", "vae_tile_size", "vae_tiling", "verbose", "width",
)  # fmt: skip

# Per adapter: the mirrored mflux main(), its first statement after the pre-load block, and the digest of everything
# from there on (load, callback registration, the per-seed generate/save loop, the except/finally). _core.run mirrors
# that tail; a new adapter adds its row here.
POST_LOAD = {
    "mflux-generate-z-image-teacache": (z_image_generate.main, "model = ZImageCommand.load(args)", "812cb2e320c4eaee"),
}


def test_mflux_build_parser_is_the_recorded_one() -> None:
    """Bug: mflux's Z-Image parser gains or drops an option (a step-cache flag, say) and the plugin
    keeps passing through a parser nobody checked against its refusals."""
    actual = ast_fingerprint(z_image_generate.build_parser)
    assert actual == BUILD_PARSER_DIGEST, f"z_image_generate.build_parser changed: review it, then record {actual}"


def test_mflux_parser_options_are_the_recorded_ones() -> None:
    """Bug: mflux adds an option through a shared parser helper (a step-cache or TeaCache-style flag on Z-Image, say);
    build_parser's own digest does not move, and the plugin passes the flag through without a refusal."""
    parser = z_image_generate.build_parser()
    strings = tuple(sorted(option for action in parser._actions for option in action.option_strings))
    dests = tuple(sorted({action.dest for action in parser._actions}))
    assert (strings, dests) == (OPTION_STRINGS, DESTS), (
        f"z_image_generate.build_parser() options changed: review them, then record\n{strings}\n{dests}"
    )


@pytest.mark.parametrize("command", sorted(POST_LOAD))
def test_the_post_load_flow_each_adapter_mirrors_is_unchanged(command: str) -> None:
    """Bug: mflux changes what its command does after load() (another callback, a different save call, a new except
    clause) and _core.run keeps mirroring the old flow; this names the tail even when a pre-load edit also moved
    MAIN_DIGEST."""
    main, first_statement, digest = POST_LOAD[command]
    actual = statements_fingerprint_from(main, first_statement)
    assert actual == digest, f"{command}: mflux main() changed after load(): review _core.run, then record {actual}"


def test_mflux_main_is_the_recorded_one() -> None:
    """Bug: mflux's main() gains a pre-load step (or Command.prepare) or changes what happens after
    load(), and the copy in z_image.py and _core.run quietly stop matching the plain command."""
    actual = ast_fingerprint(z_image_generate.main)
    assert actual == MAIN_DIGEST, f"z_image_generate.main changed: review it, then record {actual}"


def test_mflux_generate_image_is_the_recorded_one() -> None:
    """Bug: mflux's ZImage.generate_image changes what it records or how it resolves guidance, the scheduler or
    generation_parameters, and FakeZImage.generate_image keeps building images the old way, so the sidecar tests here
    pass against a record the real command no longer writes."""
    actual = ast_fingerprint(ZImage.generate_image)
    assert actual == GENERATE_IMAGE_DIGEST, (
        f"ZImage.generate_image changed: re-review FakeZImage.generate_image in tests/_fakes.py against it, "
        f"then record {actual}"
    )


def test_copied_preload_statements_match_mflux_main() -> None:
    """Bug: the copy in z_image.py drifts from mflux's main() (a dropped or changed warning branch),
    so the plugin no longer behaves like mflux-generate-z-image before load."""
    mflux_statements = set(function_body_dumps(z_image_generate.main))
    for copied in (z_image.warn_ineffective_options,):
        dumps = function_body_dumps(copied)
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


def test_load_returns_the_pipeline_the_adapter_names() -> None:
    """Bug: mflux's load() starts returning another class, and match_variant is asked about the wrong
    pipeline, so the pre-load check accepts or refuses the wrong models."""
    assert inspect.signature(ZImageCommand.load).return_annotation is ZImage
    assert z_image.PIPELINE is ZImage
    assert z_image.ADAPTER.pipeline is ZImage
