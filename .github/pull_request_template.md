## What

<!-- One paragraph: what changes and why. Name the plugin folder. -->

## Checklist

- [ ] One plugin per pull request (a `template/` change that it needs may ride along).
- [ ] Tests, ruff and ty pass from the plugin's folder.
- [ ] `uv run --python 3.13 --no-project scripts/check_plugins.py` passes from the repository root.
- [ ] The plugin's `README.md` and `CHANGELOG.md` are updated.
- [ ] A drift-test digest changed only after I read mflux's diff. mflux commit: `<sha>`

For a new plugin:

- [ ] Proposal issue: #
- [ ] It meets the acceptance criteria in `CONTRIBUTING.md`.

## Verification

<!-- The commands you ran and what they printed. -->
