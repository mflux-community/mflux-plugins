"""The footprint table in the README's "Memory at 1024×1024" section, and the committed summary next to the records,
against the committed records in benchmarks/preview-memory-2026-11-17/, read and summarized by
scripts/bench_preview_memory.py's own code."""

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bench_preview_memory.py"
RUN = ROOT / "benchmarks" / "preview-memory-2026-11-17"
README = ROOT / "README.md"
HEADING = "## Memory at 1024×1024"
TABLE_HEADER = "| Model | Setting | Plain command | TAEF command | Preview adds |"
# The table's rows, in order: the Model and Setting cells as the README prints them, and the run's model and setting.
ROWS = [
    ("Turbo", "no flag", "turbo", "default"),
    ("Turbo", "`--low-ram`", "turbo", "low-ram"),
    ("Turbo", "`--mlx-cache-limit-gb 1`", "turbo", "cache-1"),
    ("Base", "no flag", "base", "default"),
    ("Base", "`--low-ram`", "base", "low-ram"),
    ("Base", "`--mlx-cache-limit-gb 1`", "base", "cache-1"),
]
GiB = 1024**3


def _script() -> ModuleType:
    """The measuring script as a module; the one test_bench_preview_memory.py loaded, when it did."""
    loaded = sys.modules.get("bench_preview_memory")
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location("bench_preview_memory", SCRIPT)
    assert spec is not None and spec.loader is not None, f"cannot load {SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _section() -> list[str]:
    """The lines of the README's section, from below its heading to the next "## " heading."""
    lines = README.read_text(encoding="utf-8").splitlines()
    assert HEADING in lines, f"no {HEADING!r} heading in the README"
    section = lines[lines.index(HEADING) + 1 :]
    return section[: next((i for i, line in enumerate(section) if line.startswith("## ")), len(section))]


def _table() -> list[list[str]]:
    """The cells of each body row of the first table under the section's heading."""
    section = _section()
    assert TABLE_HEADER in section, f"no table with the header {TABLE_HEADER!r} under {HEADING!r}"
    start = section.index(TABLE_HEADER) + 2  # skip the header and the |---| line
    body = []
    for line in section[start:]:
        if not line.startswith("|"):
            break
        body.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return body


def test_every_readme_footprint_equals_the_median_of_the_committed_records() -> None:
    """Bug: a README cell is typed by hand and differs from the records by a hundredth (5.63 for 5.64), a row's
    plain and TAEF cells are swapped, a row sits under another setting's label, or "Preview adds" is computed from
    the rounded cells instead of the medians."""
    script = _script()
    arms = script.summarize(script.load_records(RUN))
    deltas = script.taef_minus_plain(arms)
    expected = [
        [
            model_cell,
            setting_cell,
            f"{arms[f'{model}-{setting}-plain']['footprint'] / GiB:.2f}",
            f"{arms[f'{model}-{setting}-taef']['footprint'] / GiB:.2f}",
            f"{deltas[(model, setting)]['footprint'] / GiB:+.2f}",
        ]
        for model_cell, setting_cell, model, setting in ROWS
    ]
    assert _table() == expected


def test_the_committed_summary_is_what_the_summarizer_prints_for_the_committed_records() -> None:
    """Bug: summary.txt is edited by hand, or a record is added, dropped or changed without the summary being made
    again, so the file the README cites no longer matches the records beside it."""
    script = _script()
    printed = script.format_summary(script.load_records(RUN)) + "\n"  # --summarize prints it with print()
    assert (RUN / "summary.txt").read_text(encoding="utf-8") == printed


def _paragraphs() -> list[str]:
    """The section's paragraphs (runs of non-blank lines), each with its whitespace collapsed to single spaces: a line
    break inside a paragraph renders as a space, so "mlx-taef" at a line's end and "0.9.0" on the next read
    "mlx-taef 0.9.0"."""
    text = "\n".join(_section())
    return [re.sub(r"\s+", " ", block).strip() for block in re.split(r"\n\s*\n", text) if block.strip()]


def test_the_readme_caption_links_the_summarys_commit_and_names_its_versions() -> None:
    """Bug: the caption's link shows another commit's short form than the one it points to (or than the records were
    made at), or the caption drops a version the summary lists (mlx 0.32.2 for 0.32.3) while another paragraph of the
    section still names it, so a reader sets up a different script or library to reproduce the table."""
    lines = (RUN / "summary.txt").read_text(encoding="utf-8").splitlines()
    commit = next(line for line in lines if line.startswith("Commit: ")).removeprefix("Commit: ")
    versions = next(line for line in lines if line.startswith("Versions: ")).removeprefix("Versions: ").split(", ")
    link = f"[`{commit[:7]}`](https://github.com/mflux-community/mflux-plugins/commit/{commit})"
    captions = [paragraph for paragraph in _paragraphs() if link in paragraph]
    assert len(captions) == 1, f"no single paragraph of the section holds the link {link}"
    caption = captions[0].lower()
    assert [version for version in versions if version.lower() not in caption] == []
