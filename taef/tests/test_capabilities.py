"""The record mflux-capabilities publishes for each plugin command."""

import pytest
from mflux.cli import capabilities

# (plugin command and module, the plain mflux command and module, the plugin's description)
COMMANDS = [
    pytest.param(
        ("mflux-generate-z-image-taef", "mflux.extras.taef.z_image"),
        ("mflux-generate-z-image", "mflux.models.z_image.cli.z_image_generate"),
        "Generate an image using Z-Image with live TAEF previews.",
        id="base",
    ),
    pytest.param(
        ("mflux-generate-z-image-turbo-taef", "mflux.extras.taef.z_image_turbo"),
        ("mflux-generate-z-image-turbo", "mflux.models.z_image.cli.z_image_turbo_generate"),
        "Generate an image using Z-Image Turbo with live TAEF previews.",
        id="turbo",
    ),
]
# The options a plugin command adds to its mflux command's parser.
PLUGIN_FLAGS = ("--taef-preview", "--taef-every", "--taef-frames")


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
def test_the_parser_is_mflux_parser_plus_the_plugin_options(plugin, plain, description) -> None:
    """Bug: the plugin builds its own parser, drops an mflux option, or forgets to re-export the option constants of
    its mflux command (CONDITIONAL_OPTIONS, or Turbo's IGNORED_OPTIONS), so --negative-prompt or --guidance is
    published as honored."""
    plugin_record = capabilities.describe_command(*plugin)
    plain_record = capabilities.describe_command(*plain)
    others = [option for option in plugin_record["options"] if option["flag"] not in PLUGIN_FLAGS]
    assert others == plain_record["options"]
    assert plugin_record["traits"] == plain_record["traits"]


def option_record(command: tuple[str, str], flag: str) -> dict:
    (record,) = [o for o in capabilities.describe_command(*command)["options"] if o["flag"] == flag]
    return {key: record[key] for key in ("type", "parser_default", "status")}


@pytest.mark.parametrize(("plugin", "plain", "description"), COMMANDS)
def test_the_taef_options_publish_plain_types(plugin, plain, description) -> None:
    """Bug: a preview flag uses a custom converter (published under its function name, outside mflux's wire
    vocabulary), a default that hides "not set" (--taef-every 1 instead of None), or a flag the record marks as
    anything but honored; or --taef-frames is not a plain on/off switch like mflux's own --low-ram."""
    assert option_record(plugin, "--taef-preview") == {"type": "str", "parser_default": None, "status": "honored"}
    assert option_record(plugin, "--taef-every") == {"type": "int", "parser_default": None, "status": "honored"}
    low_ram = option_record(plain, "--low-ram")
    assert option_record(plugin, "--taef-frames") == {
        "type": low_ram["type"],
        "parser_default": False,
        "status": "honored",
    }


@pytest.mark.parametrize(("plugin", "plain", "description"), COMMANDS)
def test_the_plugin_flags_are_exactly_the_three_preview_flags(plugin, plain, description) -> None:
    """Bug: an option the plugin adds is left out of PLUGIN_FLAGS (the template's example option, say), so the parser
    comparison above filters it away and nobody notices it is still published."""
    added = {o["flag"] for o in capabilities.describe_command(*plugin)["options"]} - {
        o["flag"] for o in capabilities.describe_command(*plain)["options"]
    }
    assert added == {"--taef-preview", "--taef-every", "--taef-frames"}
