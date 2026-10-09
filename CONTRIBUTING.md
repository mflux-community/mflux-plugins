# Contributing

## How a plugin folder is laid out

Each plugin is a self-contained uv project in its own folder: `pyproject.toml`, `uv.lock`, `LICENSE`, `README.md`,
`AGENTS.md`, and `NOTICE` when the plugin copies code from mflux, plus `src/mflux/extras/<name>/` and `tests/`. The
package ships only `mflux/extras/<name>/`. Never add `src/mflux/__init__.py` or `src/mflux/extras/__init__.py`; see
[mflux PR #776](https://github.com/mflux-community/mflux/pull/776) for why. There is no root `pyproject.toml`: each
plugin resolves its own dependencies and keeps its own lock.

If you add a plugin, add a `/<folder>/ @<your-handle>` line to `.github/CODEOWNERS`, so pull requests for that
folder ask you for review. The `* @IonDen` line stays; it covers the files at the repository root.

A plugin is a thin layer. It parses mflux's options, calls mflux's own command steps and hands the loaded model to
the library it wraps. The algorithm and its tests live in that library.

## Pull requests

- One plugin per pull request. Run the plugin's tests, ruff and ty from its folder first.
- Update the plugin's `README.md` with any change to a command, option or behavior.

## Releases

Each plugin is released on its own, from a tag named after its folder: `teacache-v0.1.0` publishes `mflux-teacache`
0.1.0 from `teacache/`. The version lives in the plugin's `pyproject.toml`.

A full guide to writing a plugin for mflux will be added to this file.
