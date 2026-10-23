---
name: mflux-plugin-pr
description: Use when the developer asks to push or "open the PR" in mflux-plugins, to write, fill or update a "PR description", or to "release" a plugin, "bump the version" or push a "tag" (including approving its PyPI publish).
---

# mflux-plugins pull requests

Take a change from the working tree to an open pull request, and a plugin release from the version bump to the tag.
The rules live in the repository's docs; this skill gives the order and the points where you stop and wait.

## Never without the developer's word

Never take any of these actions without the developer's explicit word in this conversation: push, force-push, create
or push a tag, merge, approve a review, approve or re-run a deployment (the `pypi-<folder>` environment publishes to
PyPI), create or edit an issue or PR, change any repository, environment, ruleset or PyPI setting, and the admin
steps in `CONTRIBUTING.md` "For maintainers: a new plugin folder". "Just push it" counts as the word for the push
only, and only after you have shown any red check (below). A word covers the action it names, given at the moment
you ask: choosing a fix you offered is not the word to push, and an earlier "then tag it" is not the word to tag.
Red checks: when tests, ruff, ty or a checker are red, show the developer the failing line, offer the fix, and do
not push unless the developer says so after seeing it; the PR's Verification section then quotes the red result as
printed. Never write a result you did not see.

## What each word covers

- Read-only calls need no word: `git status`, `git log`, `git diff`, `git fetch`, `gh pr view`, `gh pr checks`,
  `gh run view`.
- `approve` approves the PR body as you re-read it from the file, nothing more. Pushing the branch and creating the
  PR each still need the developer's word. One word covers both only when the developer says so explicitly ("approve,
  push and open it").
- This skill's own refusals override the list above; no word unlocks them. Any merge is refused, for every PR, even
  on the developer's word, even when approved and green, with or without the admin bypass. A maintainer merges on
  GitHub. Approving the deployment and re-running the release workflow are refused too (`release.md` says why and
  where the developer clicks); offer a read-only status instead.

## Steps

1. Where you start: `git status`, `git branch --show-current`, `git fetch origin`. Compare against `origin/main`.
2. Classify with `git diff --stat origin/main...HEAD` plus the uncommitted files. Plugin PR: one plugin folder
   changed (a `template/` change it needs may ride along); more than one means split it. Repo-level PR: no plugin
   folder changed (`template/`, `.github/`, `scripts/`, skills and root files only).
3. Run the checks on the working tree, before any commit. Plugin PR: as `CONTRIBUTING.md` "Pull requests" says,
   tests, ruff and ty from the plugin's folder (`template/AGENTS.md` "Commands"), and from the root
   `uv run --python 3.13 --no-project scripts/check_plugins.py`. Repo-level PR: every `run:` step of the `repo` job
   in `.github/workflows/lint.yml` (ruff, actionlint, the LICENSE compare, both checkers, the scripts tests), plus
   tests, ruff and ty in `template/` when it changed. Any PR: `scripts/check_skills.py` when `.cursor/skills/` or
   `.claude/skills/` changed. Keep each command and what it printed; Verification quotes them.
4. Any red result: stop, apply the red-check rule, and commit nothing until the developer has decided (fix it, or
   go ahead with it red).
5. Commit. On `main`, create the branch first (`git switch -c <type>/<slug>`, type `feat`, `fix`, `ci`, `chore` or
   `docs`), then commit there; never on `main`. Messages are plain: a subject and an optional body, no trailers.
6. Draft the body in `tmp-PR-content.md` by the flow in `pr-body.md`, then ask for `approve`.
7. After `approve`: re-read the file, then show the commits (`git log --oneline origin/main..HEAD`), the branch and
   the exact push and create commands, and wait for the word.
8. On a word that covers both, push (`git push -u origin <branch>`) and create the PR from the file (`pr-body.md` has
   the command). On a word for the push only, push, then ask before you create the PR. Give the developer the PR's
   URL, then remove the two draft files.
9. A release (version bump, CHANGELOG move, tag, publish): follow `release.md`.

## Body rules (details in `pr-body.md`)

- Facts only from `git log`, `git diff` against `origin/main` and, for an open PR, `gh pr view`. Never invent a
  number, a result or a reason. `What` is one plain paragraph naming the plugin folder, or the root files changed.
- A plugin PR that changes a command, option or behavior updates the plugin's `README.md` and adds a line under
  `## [Unreleased]` in its `CHANGELOG.md`. A change to docs or tests only needs no CHANGELOG line; do not ask.
- Tick a box only when a command you ran shows it holds. Boxes that state the developer's own judgment (the drift
  digest, the acceptance criteria) are left for the developer. Never add notes such as "N/A" to template lines.
- Any `tests/test_mflux_drift.py` digest changed, `template/` included: fill the `mflux commit:` line from the
  "Recorded against mflux …" comment, which must have changed with the digest. If it did not, ask; never infer.
- Repo-level PR: Verification opens with "No plugin folder changed." The title is read from the file by a command,
  never typed into the shell (backticks, `$(...)`), and the body goes in with `--body-file` only.

## Common mistakes

| Mistake | Instead |
|---|---|
| Commit, push and create in one go after the checks pass | Show the commits and commands, wait for the word |
| Commit a red tree, then ask | Check first; commit after the developer decides |
| Resend the draft you wrote before `approve` | Re-read the file; the developer may have edited it |
| Skip `uv lock` because only the version changed | `release.md`: the lock records the plugin's own version |
| Verification shows only green output after an override | Quote the red result as printed |
