# Contributing

You can contribute in two ways: change an existing plugin, or add a new one. This page covers the process. How to
write the code, from names to releases, is in [docs/writing-a-plugin.md](docs/writing-a-plugin.md).

## Report a problem

First run the plain mflux command with the same options. If it fails too, the problem is in mflux, so report it in
[mflux's issues](https://github.com/mflux-community/mflux/issues). If only the plugin's command fails, open an issue
in this repository.

## Propose a plugin

Open an issue with the ["Propose a plugin"][propose] form. Wait for a maintainer's yes before you open a pull
request, so you don't build something that can't be accepted.

A new plugin is accepted when it meets all of these:

1. It is a thin layer. The algorithm lives in a separate library on PyPI, with its own tests and releases.
2. It extends mflux commands that have command steps
   ([listed in the guide](docs/writing-a-plugin.md#which-mflux-commands-can-host-a-plugin)), and it reuses their
   `build_parser()` and steps. Any code it copies from mflux is listed in `NOTICE` and covered by a drift test, a
   test that fails when mflux changes the original.
3. Before any weights load, it refuses what it can't run, including a combination with an mflux feature it
   overlaps.
4. It starts from `template/`. The template's tests pass in renamed form, no test downloads weights, warnings are
   errors, and every test's docstring starts with `Bug:`.
5. `scripts/check_plugins.py` passes.
6. A named maintainer has the folder's line in `.github/CODEOWNERS` and answers its issues. mflux's maintainers are
   not expected to support it.
7. Its README covers install, usage and the mflux versions it works with, and it has a `CHANGELOG.md` (reviewers
   check this; the checker does not).
8. Any speed or quality number it states comes from a reproducible benchmark committed in the library it wraps.

## How a plugin folder is laid out

Each plugin is a self-contained uv project in its own folder: `pyproject.toml`, `uv.lock`, `LICENSE`, `README.md`,
`CHANGELOG.md`, `AGENTS.md`, and `NOTICE` when the plugin copies code from mflux, plus `src/mflux/extras/<folder>/`
and `tests/`. There is no root `pyproject.toml`: each plugin resolves its own dependencies and keeps its own lock.
The package ships only `mflux/extras/<folder>/`; the guide's
[What a plugin is](docs/writing-a-plugin.md#what-a-plugin-is) explains why.

A new plugin starts as a renamed copy of `template/`. The steps are under "Start a plugin" in
[template/README.md](template/README.md). The folder name sets every other name. The rules and a table
are in the guide's [Names](docs/writing-a-plugin.md#names) section.

If you add a plugin, add a `/<folder>/ @<your-handle>` line to `.github/CODEOWNERS` and a row to the Plugins table in
the root `README.md`. The checker fails without either. The CODEOWNERS line makes pull requests for that folder ask
you for review. The `* @IonDen` line stays: it is the default owner of every file that no later line matches.

`scripts/check_plugins.py` checks each plugin folder's layout, names, license, mflux range and workflows. Run it from
the repository root:

```bash
uv run --python 3.13 --no-project scripts/check_plugins.py
```

## Pull requests

- One plugin per pull request. A `template/` change that the plugin change needs rides along in that plugin's pull
  request.
- Run the plugin's tests, ruff and ty from its folder, and the checker from the repository root.
- Update the plugin's `README.md` with any change to a command, option or behavior, and add a line under
  `## [Unreleased]` in its `CHANGELOG.md`.
- Change a drift-test digest only after you have read mflux's diff and brought the plugin in line with it. Name the
  mflux commit in the pull request.
- Fill in the pull request template.

Changes reach `main` only through pull requests. A pull request merges when the `repo` check (the `lint` workflow)
passes, its plugin's `test` job is green, and a code owner other than the author approves it. When the author is
the folder's only code owner, a repository admin reviews and merges the pull request. Each plugin folder belongs to
its maintainer. `.github/`, `scripts/`, `template/` and the files at the root belong to the repository's
maintainers, so a change to a plugin's workflows needs their approval too.

## CI

The guide's [CI section](docs/writing-a-plugin.md#ci) lists the workflows and says what a red job means.

mflux's own CI does not run plugin tests, and mflux's maintainers don't keep plugins working. When mflux changes
something a plugin uses, the plugin's maintainer updates the plugin.

## Releases

Each plugin is released on its own, from a tag named after its folder: `teacache-v0.1.0` publishes `mflux-teacache`
0.1.0 from `teacache/`. The version lives in the plugin's `pyproject.toml`; bump it in a pull request. After that
pull request merges, push the tag `<folder>-v<version>` on `main`. The release workflow publishes to PyPI through
Trusted Publishing once the environment's reviewer approves. Never upload by hand. The guide's
[Releases](docs/writing-a-plugin.md#releases) section has the full steps.

## For maintainers: a new plugin folder

Before a new plugin's first tag, an org admin sets up its release:

1. Create the GitHub environment `pypi-<folder>` with a required reviewer and the deployment tag rule `<folder>-v*`.
2. Create a tag ruleset for `<folder>-v*` (one per plugin, like `teacache release tags`), so only the accounts it
   allows can push release tags.
3. On PyPI, add a pending publisher for the project `mflux-<folder>`: owner `mflux-community`, repository
   `mflux-plugins`, workflow `<folder>-release.yml`, environment `pypi-<folder>`. Without it, PyPI refuses the
   upload and the publish job fails. A pending publisher does not reserve the name, so add it shortly before the
   first tag.
4. After the first release, add a second owner to the PyPI project. Every plugin has at least two.

The CODEOWNERS line and the README row come with the plugin's pull request.

[propose]: https://github.com/mflux-community/mflux-plugins/issues/new?template=propose-a-plugin.yml
