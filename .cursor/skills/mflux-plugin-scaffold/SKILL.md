---
name: mflux-plugin-scaffold
description: Use when the developer asks to "start a plugin" or create a "new plugin" in mflux-plugins, to "scaffold" one from the template, or to "add a command" (another mflux command) to an existing plugin.
---

# mflux-plugins scaffolding

Start a plugin from `template/`, or add an mflux command to one. The rules live in the docs; this gives the order.
This skill never merges, tags or publishes; commits, the branch and the PR belong to `mflux-plugin-pr`.

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

## Ask first, then the proposal gate

Before you copy or edit anything, ask for each of these you don't have: the plugin's name, which is its folder
(rules in `docs/writing-a-plugin.md` "Names"; check the Plugins table in the root `README.md` and PyPI for
`mflux-<folder>`); the library it wraps, its PyPI name and the lower bound to require; the mflux command or
commands it extends; the maintainer's name and GitHub handle; the number of the "Propose a plugin" issue. To add a
command, ask for the plugin folder and the mflux command.

A new plugin's "Propose a plugin" issue (`.github/ISSUE_TEMPLATE/propose-a-plugin.yml`) must exist, with a
maintainer's yes, before the PR opens (`CONTRIBUTING.md` "Propose a plugin"; the PR template asks for its number).
A maintainer's own yes counts; never ask for a second approver. Without the issue, scaffold locally anyway, say the
PR waits for it, and offer to draft it from the form's fields; posting it needs the developer's word. Adding a
command to an existing plugin needs no proposal.

## New plugin

1. If `<folder>/` exists, stop and ask: a copy into it overwrites files. Otherwise copy the template from the
   repository root without its local caches (a plain `cp -R` carries them along):
   ```bash
   rsync -a --exclude='.venv*' --exclude='.ruff_cache' --exclude='.pytest_cache' \
     --exclude='__pycache__' --exclude='.DS_Store' template/ <folder>/
   ```
   This keeps `.python-version` and `CLAUDE.md`; both belong in the plugin.
2. Do every step under `template/README.md` "Start a plugin", in order, reading the list from the file (it grows;
   never work from memory or a step count). Its copy step is the command above. Fill every step from the answers:
   the names in `authors` and `maintainers`, the handle in `.github/CODEOWNERS`, and the rewrites of the README, the
   package docstring and `AGENTS.md`. When the list reaches `uv lock`, first add the library to `dependencies` next
   to `mflux`, with its lower bound (`template/AGENTS.md` "Dependencies and version pins"); if the bound is unknown or
   the library is not on PyPI yet, stop there and say so. The admin setup in `CONTRIBUTING.md` "For maintainers: a
   new plugin folder" is not yours to do.
3. A command other than Z-Image (the template ships only that one): check it under `docs/writing-a-plugin.md`
   "Which mflux commands can host a plugin", then change the touch points in `command-touch-points.md` for it.
4. Run the checks: from `<folder>/`, `uv sync --locked`, the tests, ruff and ty (`template/AGENTS.md` "Commands");
   from the root, `uv run --python 3.13 --no-project scripts/check_plugins.py`. Fix and rerun until every one is
   green and the checker counts the new folder. Show each red result as printed, with what you changed; never hide
   one or report it as green. If one stays red, stop and show it.
5. Every step above is part of the scaffold, the rewrites included. Only replacing the example option, refusal and
   `apply` hook with the library's real behavior waits for the developer (`docs/writing-a-plugin.md` "How a command
   works"). Ask before starting it.

## A red drift test while scaffolding

A fresh copy's `tests/test_mflux_drift.py` going red means mflux changed since the template was recorded: a
template defect. Follow `template/AGENTS.md` "When the drift test fails" in `template/` itself. Read mflux's diff
before you record any digest; never paste the digest the failure prints without it. Update the "Recorded against
mflux …" comment, carry the same fix into the new folder, and say so. The `template/` fix rides in the same PR.

## Adding a command

First check that the mflux command has steps (`docs/writing-a-plugin.md` "Which mflux commands can host a
plugin"); without them it can't host a plugin yet, so stop and say so. No plugin has two commands yet, the template's
tests are written for one, and a worked guide for a second command does not exist yet. Don't invent how each file
grows: change every touch point in `command-touch-points.md`, ask where the shape is unclear, then run the
checks under "New plugin" with the same rule for red results.
