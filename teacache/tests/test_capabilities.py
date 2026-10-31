"""The record mflux-capabilities publishes for each plugin command."""

import pytest
from mflux.cli import capabilities

# (plugin command and module, the plain mflux command and module, the plugin's description)
COMMANDS = [
    pytest.param(
        ("mflux-generate-z-image-teacache", "mflux.extras.teacache.z_image"),
        ("mflux-generate-z-image", "mflux.models.z_image.cli.z_image_generate"),
        "Generate an image using Z-Image with TeaCache step skipping.",
        id="z-image",
    ),
    pytest.param(
        ("mflux-generate-qwen-teacache", "mflux.extras.teacache.qwen_image"),
        ("mflux-generate-qwen", "mflux.models.qwen.cli.qwen_image_generate"),
        "Generate an image using Qwen Image model with TeaCache step skipping.",
        id="qwen",
    ),
]


@pytest.mark.parametrize(("plugin", "plain", "description"), COMMANDS)
def test_the_command_is_discovered_from_its_console_script(plugin, plain, description) -> None:
    """Bug: the console script loses the mflux-generate prefix or names another module, so
    mflux-capabilities never lists the command."""
    assert plugin in capabilities.discover_commands()


@pytest.mark.parametrize(("plugin", "plain", "description"), COMMANDS)
def test_the_record_has_full_coverage_and_its_own_description(plugin, plain, description) -> None:
    """Bug: build_parser is missing or raises (coverage runtime-only / import-error), or the plugin
    keeps mflux's description and the dump can't tell the two commands apart."""
    record = capabilities.describe_command(*plugin)
    assert record["coverage"] == "full"
    assert record["description"] == description


@pytest.mark.parametrize(("plugin", "plain", "description"), COMMANDS)
def test_the_parser_is_mflux_parser_plus_the_threshold(plugin, plain, description) -> None:
    """Bug: the plugin builds its own parser, drops an mflux option, or forgets to re-export
    CONDITIONAL_OPTIONS (so --negative-prompt is published as honored)."""
    plugin_record = capabilities.describe_command(*plugin)
    plain_record = capabilities.describe_command(*plain)
    others = [option for option in plugin_record["options"] if option["flag"] != "--teacache-threshold"]
    assert others == plain_record["options"]
    assert plugin_record["traits"] == plain_record["traits"]


@pytest.mark.parametrize(("plugin", "plain", "description"), COMMANDS)
def test_the_threshold_option_is_a_plain_float_defaulting_to_none(plugin, plain, description) -> None:
    """Bug: --teacache-threshold uses a custom converter (published under its function name, outside
    mflux's wire vocabulary) or a numeric default that hides "use the model's calibrated value"."""
    (record,) = [o for o in capabilities.describe_command(*plugin)["options"] if o["flag"] == "--teacache-threshold"]
    assert {key: record[key] for key in ("type", "parser_default", "status")} == {
        "type": "float",
        "parser_default": None,
        "status": "honored",
    }
