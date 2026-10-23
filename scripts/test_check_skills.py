"""Tests for scripts/check_skills.py. Each test builds a small repository in tmp_path that passes every rule,
breaks one rule, and checks that the checker reports exactly that problem."""

from pathlib import Path

import pytest

import check_skills

SKILL_BODY = """---
name: demo-skill
description: Use when testing.
---

# Demo

Read `CONTRIBUTING.md` "Pull requests", then `tests/_fakes.py`, then `<folder>/AGENTS.md`.
"""
SKILL_FILE = ".cursor/skills/demo-skill/SKILL.md"


def make_repo(root: Path) -> Path:
    """A repository that passes every rule: one skill, its relative link, and the docs the skill points at."""
    (root / "template" / "tests").mkdir(parents=True)
    (root / "template" / "README.md").write_text("# Template\n\n## Start a plugin\n", encoding="utf-8")
    (root / "template" / "tests" / "_fakes.py").write_text("", encoding="utf-8")
    (root / "CONTRIBUTING.md").write_text("# Contributing\n\n## Pull requests\n", encoding="utf-8")
    skill = root / ".cursor" / "skills" / "demo-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(SKILL_BODY, encoding="utf-8")
    links = root / ".claude" / "skills"
    links.mkdir(parents=True)
    (links / "demo-skill").symlink_to("../../.cursor/skills/demo-skill")
    return skill


def only_problem(root: Path) -> str:
    """The one problem the checker reports; fails when it reports none or more than one."""
    problems = check_skills.check(root)
    assert len(problems) == 1, problems
    return problems[0]


def write_skill(skill: Path, text: str) -> None:
    (skill / "SKILL.md").write_text(text, encoding="utf-8")


def test_a_valid_repo_passes(tmp_path: Path) -> None:
    """Bug: the checker reports a problem on a correct skill, so every pull request goes red."""
    make_repo(tmp_path)
    assert check_skills.check(tmp_path) == []


def test_reports_a_name_that_differs_from_the_folder(tmp_path: Path) -> None:
    """Bug: a skill folder is renamed but its name: line is not, so agents list it under a name that has no folder."""
    skill = make_repo(tmp_path)
    write_skill(skill, SKILL_BODY.replace("name: demo-skill", "name: other-skill"))
    assert only_problem(tmp_path) == "demo-skill: name 'other-skill' must equal the folder name"


def test_reports_a_name_with_capitals(tmp_path: Path) -> None:
    """Bug: a name with capitals or underscores passes, and agents that need lowercase hyphenated names skip it."""
    skill = make_repo(tmp_path)
    write_skill(skill, SKILL_BODY.replace("name: demo-skill", "name: Demo_Skill"))
    assert (
        only_problem(tmp_path)
        == "demo-skill: name 'Demo_Skill' must be lowercase letters and digits, joined by hyphens"
    )


def test_reports_an_empty_description(tmp_path: Path) -> None:
    """Bug: a skill with an empty description passes, and no agent can tell when to load it."""
    skill = make_repo(tmp_path)
    write_skill(skill, SKILL_BODY.replace("description: Use when testing.", "description:"))
    assert only_problem(tmp_path) == "demo-skill: SKILL.md frontmatter needs a non-empty description"


def test_reports_a_missing_link(tmp_path: Path) -> None:
    """Bug: a skill without its .claude/skills link passes, and Claude Code never sees the skill Cursor does."""
    make_repo(tmp_path)
    (tmp_path / ".claude" / "skills" / "demo-skill").unlink()
    assert (
        only_problem(tmp_path)
        == "demo-skill: add the link .claude/skills/demo-skill -> ../../.cursor/skills/demo-skill"
    )


def test_reports_an_absolute_link(tmp_path: Path) -> None:
    """Bug: an absolute link passes because it resolves on this machine, and it dangles in every other checkout."""
    make_repo(tmp_path)
    link = tmp_path / ".claude" / "skills" / "demo-skill"
    link.unlink()
    link.symlink_to(tmp_path / ".cursor" / "skills" / "demo-skill")
    assert (
        only_problem(tmp_path) == "demo-skill: .claude/skills/demo-skill must be a relative link, not an absolute path"
    )


def test_reports_a_dangling_link(tmp_path: Path) -> None:
    """Bug: a link whose target folder is gone passes, and the agent loads nothing."""
    make_repo(tmp_path)
    (tmp_path / ".claude" / "skills" / "ghost").symlink_to("../../.cursor/skills/ghost")
    assert only_problem(tmp_path) == ".claude/skills/ghost: the link points at nothing"


def test_reports_a_real_directory_under_links(tmp_path: Path) -> None:
    """Bug: a copied folder under .claude/skills passes, and it drifts from the .cursor/skills original."""
    make_repo(tmp_path)
    (tmp_path / ".claude" / "skills" / "stray").mkdir()
    assert (
        only_problem(tmp_path)
        == ".claude/skills/stray: must be a symlink into .cursor/skills, not a real file or folder"
    )


def test_reports_a_real_directory_with_a_skill_name_under_links(tmp_path: Path) -> None:
    """Bug: a copied folder that takes a skill's own name passes, and it drifts from the .cursor/skills original."""
    make_repo(tmp_path)
    link = tmp_path / ".claude" / "skills" / "demo-skill"
    link.unlink()
    link.mkdir()
    assert (
        only_problem(tmp_path)
        == ".claude/skills/demo-skill: must be a symlink into .cursor/skills, not a real file or folder"
    )


def test_reports_a_link_to_the_wrong_skill(tmp_path: Path) -> None:
    """Bug: a skill's relative link points at another skill's folder, so the agent loads the wrong instructions."""
    make_repo(tmp_path)
    other = tmp_path / ".cursor" / "skills" / "other-skill"
    other.mkdir()
    write_skill(other, "---\nname: other-skill\ndescription: Use when testing too.\n---\n")
    links = tmp_path / ".claude" / "skills"
    (links / "other-skill").symlink_to("../../.cursor/skills/other-skill")
    (links / "demo-skill").unlink()
    (links / "demo-skill").symlink_to("../../.cursor/skills/other-skill")
    assert (
        only_problem(tmp_path) == "demo-skill: .claude/skills/demo-skill must point at ../../.cursor/skills/demo-skill"
    )


def test_reports_a_missing_skill_md(tmp_path: Path) -> None:
    """Bug: a skill folder without SKILL.md passes, and agents find a folder with nothing to load."""
    skill = make_repo(tmp_path)
    (skill / "SKILL.md").unlink()
    assert only_problem(tmp_path) == "demo-skill: add .cursor/skills/demo-skill/SKILL.md"


def test_reports_a_skill_md_without_frontmatter(tmp_path: Path) -> None:
    """Bug: a SKILL.md with no frontmatter passes, and agents cannot read its name or description."""
    skill = make_repo(tmp_path)
    write_skill(skill, "# Demo\n")
    assert only_problem(tmp_path) == "demo-skill: SKILL.md must start with a --- frontmatter block"


def test_reports_frontmatter_without_a_name(tmp_path: Path) -> None:
    """Bug: a frontmatter block without name: passes, and agents list the skill under no name at all."""
    skill = make_repo(tmp_path)
    write_skill(skill, SKILL_BODY.replace("name: demo-skill\n", ""))
    assert only_problem(tmp_path) == "demo-skill: SKILL.md frontmatter needs a name"


def test_reports_an_orphan_link(tmp_path: Path) -> None:
    """Bug: a link left behind after its skill was renamed or removed passes, and the agent lists a dead skill."""
    make_repo(tmp_path)
    (tmp_path / ".claude" / "skills" / "extra").symlink_to("../../.cursor/skills/demo-skill")
    assert only_problem(tmp_path) == ".claude/skills/extra: no skill folder .cursor/skills/extra"


def test_reports_a_missing_path(tmp_path: Path) -> None:
    """Bug: a skill cites a file that was renamed or deleted and the pointer rots without a red check."""
    skill = make_repo(tmp_path)
    write_skill(skill, SKILL_BODY + "Also see `docs/gone.md`.\n")
    assert only_problem(tmp_path) == f"{SKILL_FILE}: `docs/gone.md` does not exist"


def test_reports_a_missing_heading(tmp_path: Path) -> None:
    """Bug: a skill cites a section that was renamed in CONTRIBUTING.md and the pointer now leads nowhere."""
    make_repo(tmp_path)
    (tmp_path / "CONTRIBUTING.md").write_text("# Contributing\n\n## Merging\n", encoding="utf-8")
    assert only_problem(tmp_path) == f'{SKILL_FILE}: `CONTRIBUTING.md` has no heading "Pull requests"'


@pytest.mark.parametrize(
    "citation",
    ['`CONTRIBUTING.md`\n   "Missing section"', '`CONTRIBUTING.md` "Missing\n   section"'],
    ids=["break-before-the-quote", "break-inside-the-quotes"],
)
def test_reports_a_missing_heading_cited_across_a_line_break(tmp_path: Path, citation: str) -> None:
    """Bug: a heading citation that the editor wrapped onto the next line is never checked, so it rots silently."""
    skill = make_repo(tmp_path)
    write_skill(skill, SKILL_BODY + f"Then read {citation}.\n")
    assert only_problem(tmp_path) == f'{SKILL_FILE}: `CONTRIBUTING.md` has no heading "Missing section"'


def test_accepts_a_heading_cited_across_a_line_break(tmp_path: Path) -> None:
    """Bug: the wrapped heading text is compared with its line break and indent, so a correct citation is reported."""
    skill = make_repo(tmp_path)
    write_skill(skill, SKILL_BODY + 'Then read `CONTRIBUTING.md` "Pull\n   requests".\n')
    assert check_skills.check(tmp_path) == []


def test_reports_a_missing_path_in_another_file_of_the_skill(tmp_path: Path) -> None:
    """Bug: only SKILL.md is read, so a dead pointer in the skill's pr-body.md never turns the check red."""
    skill = make_repo(tmp_path)
    (skill / "pr-body.md").write_text("See `docs/gone.md`.\n", encoding="utf-8")
    assert only_problem(tmp_path) == ".cursor/skills/demo-skill/pr-body.md: `docs/gone.md` does not exist"


def test_ignores_paths_inside_fenced_code_blocks(tmp_path: Path) -> None:
    """Bug: a path inside an example shell block is checked, so a sample command that creates a file fails the skill."""
    skill = make_repo(tmp_path)
    write_skill(skill, SKILL_BODY + "\n```sh\necho `docs/gone.md`\n```\n")
    assert check_skills.check(tmp_path) == []


def test_resolves_paths_under_template(tmp_path: Path) -> None:
    """Bug: tests/_fakes.py is looked up only at the repo root, so a path that lives in template/ is reported missing."""
    make_repo(tmp_path)
    assert not (tmp_path / "tests").exists()
    assert check_skills.check(tmp_path) == []


def test_resolves_paths_in_the_skill_folder(tmp_path: Path) -> None:
    """Bug: a skill that cites its own sibling file, pr-body.md, is told the file does not exist."""
    skill = make_repo(tmp_path)
    (skill / "pr-body.md").write_text("# Body\n", encoding="utf-8")
    write_skill(skill, SKILL_BODY + "Fill in `pr-body.md`.\n")
    assert check_skills.check(tmp_path) == []


def test_skips_gitignored_paths(tmp_path: Path) -> None:
    """Bug: a file the procedure creates and git ignores, tmp-PR-content.md, is reported missing on a clean checkout."""
    skill = make_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("# scratch\ntmp-PR-content.md\n", encoding="utf-8")
    write_skill(skill, SKILL_BODY + "Write the body to `tmp-PR-content.md`.\n")
    assert check_skills.check(tmp_path) == []


def test_reports_a_missing_path_that_a_gitignore_negation_re_includes(tmp_path: Path) -> None:
    """Bug: `!` lines are dropped, so `.claude/*` hides every dead `.claude/skills/...` pointer despite
    `!.claude/skills/`."""
    skill = make_repo(tmp_path)
    (tmp_path / ".gitignore").write_text(".claude/*\n!.claude/skills/\n", encoding="utf-8")
    write_skill(skill, SKILL_BODY + "Link it from `.claude/skills/nope/`.\n")
    assert only_problem(tmp_path) == f"{SKILL_FILE}: `.claude/skills/nope/` does not exist"


def test_skips_a_path_under_an_ignored_directory(tmp_path: Path) -> None:
    """Bug: a `scratch/` pattern is compared with the whole path only, so `scratch/notes.md` is reported missing."""
    skill = make_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("scratch/\n", encoding="utf-8")
    write_skill(skill, SKILL_BODY + "Keep notes in `scratch/notes.md`.\n")
    assert check_skills.check(tmp_path) == []


def test_a_gitignore_star_does_not_cross_a_slash(tmp_path: Path) -> None:
    """Bug: `docs/*.md` is matched with plain fnmatch, whose * crosses /, so a dead `docs/sub/gone.md` is skipped."""
    skill = make_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("docs/*.md\n", encoding="utf-8")
    write_skill(skill, SKILL_BODY + "See `docs/sub/gone.md`.\n")
    assert only_problem(tmp_path) == f"{SKILL_FILE}: `docs/sub/gone.md` does not exist"


def test_no_skills_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Bug: a repository with no skills exits 1, so the check fails before the first skill is added."""
    assert check_skills.main([str(tmp_path)]) == 0
    assert capsys.readouterr().out == "check_skills: no skills found\n"


def test_main_prints_each_problem_and_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Bug: main() prints problems but exits 0, so the lint job stays green on a broken skill."""
    make_repo(tmp_path)
    (tmp_path / ".claude" / "skills" / "demo-skill").unlink()
    assert check_skills.main([str(tmp_path)]) == 1
    assert capsys.readouterr().out == (
        "demo-skill: add the link .claude/skills/demo-skill -> ../../.cursor/skills/demo-skill\n"
    )


def test_main_exits_0_on_a_correct_repository(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Bug: main() exits 1 on a clean repository, so every pull request fails the required check."""
    make_repo(tmp_path)
    assert check_skills.main([str(tmp_path)]) == 0
    assert capsys.readouterr().out == "check_skills: 1 skill(s) follow the rules\n"
