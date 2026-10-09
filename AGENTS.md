# AGENTS.md

Repository-wide rules for coding agents. Each plugin folder has its own `AGENTS.md` with the package's rules; read
it before you change that folder. Before you write or change a plugin, read `docs/writing-a-plugin.md`.

- One folder per plugin (`teacache/`; `template/` is the starting point for new ones and is never published), each
  an independent uv project with its own `uv.lock`. Don't add a root `pyproject.toml` or a uv workspace. Plugins wrap
  different libraries with different mflux ranges, and one shared lock would tie them together.
- Run `uv`, `pytest`, `ruff` and `ty` from the plugin's folder.
- `LICENSE` and `NOTICE` live inside each plugin folder, because packaging can't reach files above the project
  folder. The root `LICENSE` is a copy, and CI checks that the copies are identical.
- CI: `.github/workflows/lint.yml` runs on every pull request. It runs ruff over the whole tree with each plugin's
  own settings, lints the workflows, compares the license copies and runs `scripts/check_plugins.py` and its tests.
  Its ruff version equals the plugins' `ruff==` dev pin, and the checker fails the job when they differ; change them
  together. `<folder>.yml` runs when its folder changes and once a day (`template.yml`: once a week); published
  plugins also have `<folder>-release.yml`, which runs on their tags and calls `<folder>.yml` to test before it
  publishes.
- A new plugin starts as a copy of `template/`, renamed. Its folder name is lowercase letters and digits only,
  starting with a letter, and it is the plugin's name everywhere: `mflux.extras.<folder>`, `mflux-<folder>`, the
  command suffix `-<folder>`, the tags `<folder>-v*` and the environment `pypi-<folder>`.
- `scripts/check_plugins.py` checks every plugin folder's layout, names, license, mflux range and workflow wiring in
  the `lint` job. Before you push, run it from the repository root:
  `uv run --python 3.13 --no-project scripts/check_plugins.py`.
- A change to a generic part of a plugin (the pyproject tool blocks, the conftest guards, the namespace, wheel,
  isolation or drift tests, the test or release workflow) is also made in `template/`, in the same pull request.
- Releases: a tag `<folder>-v<version>` on `main`, equal to the plugin's `project.version`, publishes through
  Trusted Publishing and the GitHub environment `pypi-<folder>`. Never upload by hand. Before a new plugin's first
  tag, an org admin creates the `pypi-<folder>` environment with a required reviewer, the tag rule for `<folder>-v*`
  and the PyPI pending publisher.
- One plugin per pull request. A `template/` change that the plugin change needs rides along in that plugin's pull
  request.
