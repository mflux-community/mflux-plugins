# AGENTS.md

Repository-wide rules for coding agents. Each plugin folder has its own `AGENTS.md` with the package's rules; read
it before you change that folder.

- One folder per plugin (`teacache/`), each an independent uv project with its own `uv.lock`. Don't add a root
  `pyproject.toml` or a uv workspace. Plugins wrap different libraries with different mflux ranges, and one shared
  lock would tie them together.
- Run `uv`, `pytest`, `ruff` and `ty` from the plugin's folder.
- `LICENSE` and `NOTICE` live inside each plugin folder, because packaging can't reach files above the project
  folder. The root `LICENSE` is a copy, and CI checks that the copies are identical.
- CI: `.github/workflows/lint.yml` runs on every pull request. It runs ruff over the whole tree with each plugin's
  own settings, lints the workflows and compares the license copies. Its ruff version equals the plugins' `ruff==`
  dev pin; change them together. Each plugin also has `<folder>.yml`, which runs when its folder changes and once a
  day, and `<folder>-release.yml`, which runs on its tags.
- Releases: a tag `<folder>-v<version>` on `main`, equal to the plugin's `project.version`, publishes through
  Trusted Publishing and the GitHub environment `pypi-<folder>`. Never upload by hand.
- One plugin per pull request.
