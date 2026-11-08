"""The fingerprints the drift test pins mflux's functions with: parser_fingerprint, which the build_parser pins use,
reads past the wording of help texts; ast_fingerprint, which every other pin uses, does not."""

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from _fingerprint import ast_fingerprint, parser_fingerprint

# A build_parser in the shape of mflux's: a parser built with prose, shared helper calls, one of them with a help
# text, and one argument spelled out with every keyword a parser option can carry.
BASE = '''
import argparse


def build_parser():
    """Builds the parser."""
    parser = argparse.ArgumentParser(description="Generate an image.", epilog="See the README.")
    parser.add_general_arguments()
    parser.add_image_to_image_arguments(required=False)
    parser.add_small_text_encoder_arguments(help_text="Downloads ~3.3 GB instead of ~7.5 GB.")
    parser.add_argument(
        "--steps", "-s", dest="steps", type=int, default=4, choices=[4, 8], action="store", nargs="?",
        metavar="N", required=False, help="Denoising steps.",
    )
    parser.add_argument("--seed", help=f"The seed, {DEFAULT_SEED} by default.")
    parser.add_output_arguments()
    parser.set_defaults(scheduler="flow_match_euler_discrete")
    return parser
'''

# Edits to BASE that change only the wording a user reads in --help, as (old, new).
WORDING_EDITS = {
    "help_text": (
        'help_text="Downloads ~3.3 GB instead of ~7.5 GB."',
        'help_text="Downloads ~3.6 GB instead of ~8.0 GB."',
    ),
    "help": ('help="Denoising steps."', 'help="How many denoising steps to run."'),
    "description": ('description="Generate an image."', 'description="Generate an image from a prompt."'),
    "epilog": ('epilog="See the README."', 'epilog="See the Z-Image README."'),
}

# Edits to BASE that change what the parser accepts or produces, as (old, new).
PARSER_EDITS = {
    "option string": ('"--steps", "-s"', '"--num-steps", "-s"'),
    "dest": ('dest="steps"', 'dest="num_steps"'),
    "default": ("default=4", "default=8"),
    "type": ("type=int", "type=float"),
    "choices": ("choices=[4, 8]", "choices=[4, 8, 16]"),
    "action": ('action="store"', 'action="append"'),
    "nargs": ('nargs="?"', 'nargs="*"'),
    "metavar": ('metavar="N"', 'metavar="STEPS"'),
    "required": ('metavar="N", required=False', 'metavar="N", required=True'),
    "helper argument": ("add_image_to_image_arguments(required=False)", "add_image_to_image_arguments(required=True)"),
    "helper added": (
        "    parser.add_output_arguments()\n",
        "    parser.add_output_arguments()\n    parser.add_float32_arguments()\n",
    ),
    "helper removed": ("    parser.add_general_arguments()\n", ""),
    "helpers reordered": (
        "    parser.add_general_arguments()\n    parser.add_image_to_image_arguments(required=False)\n",
        "    parser.add_image_to_image_arguments(required=False)\n    parser.add_general_arguments()\n",
    ),
    # A help text built from a value, not written out: the value can be an option's, so it still counts.
    "value inside an f-string help": ("{DEFAULT_SEED}", "{OTHER_SEED}"),
    # Only the wording of a prose keyword is ignored, not the keyword: a helper that takes no help_text, or takes it
    # under another name, fails when the parser is built.
    "prose keyword added": ("add_output_arguments()", 'add_output_arguments(help_text="x")'),
    "prose keyword renamed": ('help_text="Downloads ~3.3', 'help="Downloads ~3.3'),
    # A parser default is a value the command runs with, like mflux's set_defaults(scheduler=...).
    "set_defaults value": ('scheduler="flow_match_euler_discrete"', 'scheduler="flow_match_heun_discrete"'),
}


def build_parser_from(source: str, folder: Path, name: str) -> Callable[..., Any]:
    """The build_parser of `source`, loaded from a file so that inspect.getsource reads it as it reads mflux's."""
    path = folder / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"fingerprint_case_{folder.name}_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_parser


def edited(old: str, new: str) -> str:
    # A fragment BASE doesn't hold would leave the source unchanged and every comparison vacuous.
    assert BASE.count(old) == 1, f"BASE holds {old!r} {BASE.count(old)} times"
    return BASE.replace(old, new)


@pytest.mark.parametrize("keyword", sorted(WORDING_EDITS))
def test_a_help_wording_edit_leaves_the_parser_fingerprint_unchanged(keyword: str, tmp_path: Path) -> None:
    """Bug: parser_fingerprint still reads the text of help, help_text, description or epilog, so an mflux commit
    that only rewords a --help line (the Z-Image Turbo --small-text-encoder download sizes) moves a build_parser pin
    and the drift test asks for a review of a parser that accepts exactly what it did."""
    before = build_parser_from(BASE, tmp_path, "before")
    after = build_parser_from(edited(*WORDING_EDITS[keyword]), tmp_path, "after")
    assert parser_fingerprint(after) == parser_fingerprint(before)


def test_a_docstring_edit_leaves_the_parser_fingerprint_unchanged(tmp_path: Path) -> None:
    """Bug: parser_fingerprint stops stripping docstrings, and a reworded docstring moves a build_parser pin."""
    before = build_parser_from(BASE, tmp_path, "before")
    after = build_parser_from(
        edited('"""Builds the parser."""', '"""Builds the command line parser."""'), tmp_path, "after"
    )
    assert parser_fingerprint(after) == parser_fingerprint(before)


@pytest.mark.parametrize("edit", sorted(PARSER_EDITS))
def test_a_parser_edit_changes_the_parser_fingerprint(edit: str, tmp_path: Path) -> None:
    """Bug: parser_fingerprint drops more than the help texts (every string constant, every keyword argument, the
    helper calls' order), so mflux renames an option, changes its default or type, or adds a helper that brings a
    step-skipping flag, and no build_parser pin moves."""
    before = build_parser_from(BASE, tmp_path, "before")
    after = build_parser_from(edited(*PARSER_EDITS[edit]), tmp_path, "after")
    assert parser_fingerprint(after) != parser_fingerprint(before)


@pytest.mark.parametrize("keyword", sorted(WORDING_EDITS))
def test_a_help_wording_edit_still_changes_the_ordinary_fingerprint(keyword: str, tmp_path: Path) -> None:
    """Bug: the help-blindness leaks into ast_fingerprint, which pins every function that is not a build_parser, so a
    changed string there (a warning's text, a metadata key passed as a help= keyword) no longer moves its pin."""
    before = build_parser_from(BASE, tmp_path, "before")
    after = build_parser_from(edited(*WORDING_EDITS[keyword]), tmp_path, "after")
    assert ast_fingerprint(after) != ast_fingerprint(before)


def test_the_parser_fingerprint_leaves_the_ordinary_one_of_the_same_function_alone(tmp_path: Path) -> None:
    """Bug: parser_fingerprint blanks the help texts in a tree ast_fingerprint reads again (a cached parse of the
    function), so after the build_parser pin runs, ast_fingerprint of that function stops seeing its help texts."""
    build_parser = build_parser_from(BASE, tmp_path, "base")
    before = ast_fingerprint(build_parser)
    parser_fingerprint(build_parser)
    assert ast_fingerprint(build_parser) == before
