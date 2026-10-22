"""Check the agent skills in this repository against its rules.

Run from the repository root: uv run --python 3.13 --no-project scripts/check_skills.py
Exit 0 when every rule holds. Otherwise exit 1 and print one line per problem.
Standard library only (Python 3.11 or newer).

Each skill is a folder .cursor/skills/<name>/ with a SKILL.md, and each has a relative symlink
.claude/skills/<name> -> ../../.cursor/skills/<name>. The checker keeps them consistent:

- SKILL.md starts with a frontmatter block holding name and description; the name is lowercase words joined by
  hyphens and equals the folder name; the description is not empty.
- Every skill has its link, the link is relative and resolves to the skill folder, and every entry under
  .claude/skills is a symlink into .cursor/skills (no copied folder, no dangling link, no link without a skill).
- Pointers: every .md file in a skill folder may point at repository docs, and each pointer must still resolve.
  A pointer is an inline-code path such as `CONTRIBUTING.md` or `template/`; it must exist relative to the repository
  root, to template/, or to the skill's own folder (so a skill can cite its sibling `pr-body.md`). A path with a
  <placeholder> in it never looks like a path (PATH_TOKEN has no < or >), so it is skipped. A path the root
  .gitignore ignores is skipped too (a file a procedure creates and git ignores); the patterns follow git's rules
  closely enough for this: last match wins, ! re-includes, * stays inside one path segment, a pattern with a / before
  its end is anchored to the root, and a path under an ignored folder is ignored. A heading reference is an
  inline-code .md path, whitespace (a line break included) and a double-quoted heading, for example
  `CONTRIBUTING.md` "Pull requests"; that file must have a heading line with that text, compared with runs of
  whitespace collapsed to one space, so a heading may wrap inside the quotes. Fenced code blocks are not read."""

from __future__ import annotations

import fnmatch
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

SKILLS_DIR = Path(".cursor/skills")
LINKS_DIR = Path(".claude/skills")
NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
PATH_TOKEN = re.compile(r"^[\w.][\w./-]*\.(?:md|py|toml|yml|yaml|lock)$|^[\w.][\w./-]*/$")
CODE_SPAN = re.compile(r"`([^`\n]+)`")
HEADING_REFERENCE = re.compile(r'`([^`\n]+)`\s+"([^"]+)"')
FENCED_BLOCK = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.MULTILINE | re.DOTALL)
TEMPLATE_FOLDER = "template"


def read(path: Path) -> str:
    """A file's text, or "" when it is missing. A byte that is not UTF-8 is replaced, not fatal."""
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def frontmatter(text: str) -> dict[str, str] | None:
    """The key: value pairs between the first two --- lines, or None when the file does not start with one."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    values: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return values
        key, separator, value = line.partition(":")
        if separator and not line[:1].isspace():
            values[key.strip()] = value.strip().strip("\"'")
    return None


@dataclass(frozen=True)
class IgnoreRule:
    """One .gitignore line: its path segments, and whether it negates, matches folders only, or is anchored."""

    segments: tuple[str, ...]
    negated: bool
    folders_only: bool
    anchored: bool


def ignore_rules(root: Path) -> list[IgnoreRule]:
    """The root .gitignore's rules in file order, without comments and blank lines."""
    rules = []
    for line in read(root / ".gitignore").splitlines():
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        if negated or line.startswith("\\"):
            line = line[1:]
        folders_only = line.endswith("/")
        line = line.rstrip("/")
        anchored = "/" in line
        line = line.lstrip("/")
        if line:
            rules.append(IgnoreRule(tuple(line.split("/")), negated, folders_only, anchored))
    return rules


def segments_match(path: tuple[str, ...], pattern: tuple[str, ...]) -> bool:
    """True when the path segments match the pattern segments; ** matches any number of segments, and *, ? and [...]
    match inside one segment only."""
    if not pattern:
        return not path
    if pattern[0] == "**":
        return any(segments_match(path[start:], pattern[1:]) for start in range(len(path) + 1))
    return bool(path) and fnmatch.fnmatchcase(path[0], pattern[0]) and segments_match(path[1:], pattern[1:])


def rule_matches(rule: IgnoreRule, path: tuple[str, ...], is_folder: bool) -> bool:
    if rule.folders_only and not is_folder:
        return False
    if rule.anchored:
        return segments_match(path, rule.segments)
    return segments_match(path[-1:], rule.segments)


def is_ignored(token: str, rules: list[IgnoreRule]) -> bool:
    """True when git would ignore the path: the last rule that matches decides, and a path whose folder is ignored is
    ignored with it (git cannot re-include a file inside an excluded folder). A token ending in / is a folder."""
    parts = tuple(part for part in token.strip("/").split("/") if part)
    for end in range(1, len(parts) + 1):
        is_folder = end < len(parts) or token.endswith("/")
        verdict = False
        for rule in rules:
            if rule_matches(rule, parts[:end], is_folder):
                verdict = not rule.negated
        if verdict:
            return True
    return False


def resolve(root: Path, skill: Path, token: str) -> Path | None:
    """The existing file or folder a pointer names: under the repository root, template/ or the skill folder."""
    for base in (root, root / TEMPLATE_FOLDER, skill):
        if (base / token).exists():
            return base / token
    return None


def has_heading(path: Path, heading: str) -> bool:
    pattern = re.compile(rf"^#{{1,6}} {re.escape(heading)}\s*$")
    return any(pattern.match(line) for line in read(path).splitlines())


def check_pointers(root: Path, skill: Path, rules: list[IgnoreRule]) -> list[str]:
    problems = []
    for doc in sorted(skill.rglob("*.md")):
        where = doc.relative_to(root).as_posix()
        text = FENCED_BLOCK.sub("", read(doc))
        for token in dict.fromkeys(CODE_SPAN.findall(text)):
            if not PATH_TOKEN.match(token) or is_ignored(token, rules):
                continue
            if resolve(root, skill, token) is None:
                problems.append(f"{where}: `{token}` does not exist")
        for token, wrapped in dict.fromkeys(HEADING_REFERENCE.findall(text)):
            heading = " ".join(wrapped.split())
            if not PATH_TOKEN.match(token) or is_ignored(token, rules):
                continue
            target = resolve(root, skill, token)
            if target is not None and target.is_file() and not has_heading(target, heading):
                problems.append(f'{where}: `{token}` has no heading "{heading}"')
    return problems


def check_skill(root: Path, skill: Path, rules: list[IgnoreRule]) -> list[str]:
    problems: list[str] = []
    name = skill.name
    manifest = skill / "SKILL.md"
    if not manifest.is_file():
        problems.append(f"{name}: add {manifest.relative_to(root).as_posix()}")
    else:
        values = frontmatter(read(manifest))
        if values is None:
            problems.append(f"{name}: SKILL.md must start with a --- frontmatter block")
        else:
            declared = values.get("name")
            if declared is None:
                problems.append(f"{name}: SKILL.md frontmatter needs a name")
            elif not NAME_PATTERN.match(declared):
                problems.append(f"{name}: name {declared!r} must be lowercase letters and digits, joined by hyphens")
            elif declared != name:
                problems.append(f"{name}: name {declared!r} must equal the folder name")
            if not values.get("description"):
                problems.append(f"{name}: SKILL.md frontmatter needs a non-empty description")

    link = root / LINKS_DIR / name
    expected = f"../../{SKILLS_DIR.as_posix()}/{name}"
    if not os.path.lexists(link):
        problems.append(f"{name}: add the link {LINKS_DIR.as_posix()}/{name} -> {expected}")
    elif link.is_symlink():
        if os.path.isabs(os.readlink(link)):
            problems.append(f"{name}: {LINKS_DIR.as_posix()}/{name} must be a relative link, not an absolute path")
        elif link.resolve() != skill.resolve():
            problems.append(f"{name}: {LINKS_DIR.as_posix()}/{name} must point at {expected}")
    # A link entry that is not a symlink is reported with the other stray entries, below.
    problems.extend(check_pointers(root, skill, rules))
    return problems


def check_link(root: Path, link: Path) -> list[str]:
    """Problems with one entry under LINKS_DIR whose name has no skill folder. A skill's own link is checked with it."""
    where = link.relative_to(root).as_posix()
    if not link.is_symlink():
        return [f"{where}: must be a symlink into {SKILLS_DIR.as_posix()}, not a real file or folder"]
    if not link.exists():
        return [f"{where}: the link points at nothing"]
    if not link.resolve().is_relative_to((root / SKILLS_DIR).resolve()):
        return [f"{where}: the link must resolve into {SKILLS_DIR.as_posix()}"]
    return [f"{where}: no skill folder {SKILLS_DIR.as_posix()}/{link.name}"]


def skill_folders(root: Path) -> list[Path]:
    directory = root / SKILLS_DIR
    return sorted(path for path in directory.iterdir() if path.is_dir()) if directory.is_dir() else []


def check(root: Path) -> list[str]:
    """One line per problem; empty when every rule holds."""
    skills = skill_folders(root)
    names = {skill.name for skill in skills}
    rules = ignore_rules(root)
    problems: list[str] = []
    for skill in skills:
        problems.extend(check_skill(root, skill, rules))
    links = root / LINKS_DIR
    for link in sorted(links.iterdir()) if links.is_dir() else []:
        if link.name in names:
            if not link.is_symlink():
                problems.extend(check_link(root, link))
        else:
            problems.extend(check_link(root, link))
    return problems


def main(argv: list[str] | None = None) -> int:
    root = Path(argv[0]) if argv else Path.cwd()
    skills = skill_folders(root)
    problems = check(root)
    if not skills and not problems:
        print("check_skills: no skills found")
        return 0
    for line in problems:
        print(line)
    if not problems:
        print(f"check_skills: {len(skills)} skill(s) follow the rules")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
