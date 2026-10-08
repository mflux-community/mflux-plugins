"""The record mflux-capabilities publishes for the plugin command."""

from mflux.cli import capabilities

PLUGIN = ("mflux-generate-z-image-teacache", "mflux.extras.teacache.z_image")
PLAIN = ("mflux-generate-z-image", "mflux.models.z_image.cli.z_image_generate")


def test_the_command_is_discovered_from_its_console_script() -> None:
    """Bug: the console script loses the mflux-generate prefix or names another module, so
    mflux-capabilities never lists the command."""
    assert PLUGIN in capabilities.discover_commands()


def test_the_record_has_full_coverage_and_its_own_description() -> None:
    """Bug: build_parser is missing or raises (coverage runtime-only / import-error), or the plugin
    keeps mflux's description and the dump can't tell the two commands apart."""
    record = capabilities.describe_command(*PLUGIN)
    assert record["coverage"] == "full"
    assert record["description"] == "Generate an image using Z-Image with TeaCache step skipping."


def test_the_parser_is_mflux_parser_plus_the_threshold() -> None:
    """Bug: the plugin builds its own parser, drops an mflux option, or forgets to re-export
    CONDITIONAL_OPTIONS (so --negative-prompt is published as honored)."""
    plugin = capabilities.describe_command(*PLUGIN)
    plain = capabilities.describe_command(*PLAIN)
    others = [option for option in plugin["options"] if option["flag"] != "--teacache-threshold"]
    assert others == plain["options"]
    assert plugin["traits"] == plain["traits"]


def test_the_threshold_option_is_a_plain_float_defaulting_to_none() -> None:
    """Bug: --teacache-threshold uses a custom converter (published under its function name, outside
    mflux's wire vocabulary) or a numeric default that hides "use the model's calibrated value"."""
    (record,) = [o for o in capabilities.describe_command(*PLUGIN)["options"] if o["flag"] == "--teacache-threshold"]
    assert {key: record[key] for key in ("type", "parser_default", "status")} == {
        "type": "float",
        "parser_default": None,
        "status": "honored",
    }
