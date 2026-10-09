# Contributing

## How a plugin folder is laid out

Each plugin is a self-contained uv project in its own folder: `pyproject.toml`, `uv.lock`, `LICENSE`, `README.md`,
`AGENTS.md`, and `NOTICE` when the plugin copies code from mflux, plus `src/mflux/extras/<folder>/` and `tests/`.
The package ships only `mflux/extras/<folder>/`. Never add `src/mflux/__init__.py` or
`src/mflux/extras/__init__.py`; see [mflux PR #776](https://github.com/mflux-community/mflux/pull/776) for why.
There is no root `pyproject.toml`: each plugin resolves its own dependencies and keeps its own lock.

If you add a plugin, add a `/<folder>/ @<your-handle>` line to `.github/CODEOWNERS`, so pull requests for that
folder ask you for review. The `* @IonDen` line stays: it is the default owner of every file that no later line
matches.

A plugin is a thin layer. It parses mflux's options, calls mflux's own command steps and hands the loaded model to
the library it wraps. The algorithm and its tests live in that library.

A new plugin starts as a renamed copy of `template/`; its README lists the steps. The folder name is lowercase
letters and digits only, starting with a letter, and the plugin uses it everywhere: in its package path,
`mflux-<folder>` on PyPI, its command suffix and its tags. `scripts/check_plugins.py` checks the folder rules on this
page (layout, names, license, mflux range, workflows). Run it from the repository root:
`uv run --python 3.13 --no-project scripts/check_plugins.py`.

## Pull requests

- One plugin per pull request. A `template/` change that the plugin change needs rides along in that plugin's pull
  request. Run the plugin's tests, ruff and ty from its folder first.
- Update the plugin's `README.md` with any change to a command, option or behavior.

## Releases

Each plugin is released on its own, from a tag named after its folder: `teacache-v0.1.0` publishes `mflux-teacache`
0.1.0 from `teacache/`. The version lives in the plugin's `pyproject.toml`.

A full guide to writing a plugin for mflux will come in a later pull request to this repository and will live in
`docs/`.
