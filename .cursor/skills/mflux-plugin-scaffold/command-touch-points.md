# Touch points for an mflux command

What a plugin changes for each mflux command it extends: a command added to a plugin, or a new plugin whose command
is not the template's Z-Image. No worked guide for a second command exists yet; where a file's shape for two
commands is unclear, ask instead of inventing it.

- `src/mflux/extras/<folder>/<command>.py`: the command's adapter module, like the Z-Image one.
- An entry in `[project.scripts]` of `pyproject.toml`, named by `docs/writing-a-plugin.md` "Names", and the
  entry-point assertion in `tests/test_wheel.py`.
- The fakes in `tests/_fakes.py` for the command's model class.
- The drift pins in `tests/test_mflux_drift.py` for the command's parser, `main()` and model steps.
- `tests/test_capabilities.py` and the option constants it reads, re-exported from the mflux command's module.
- The model guard in `tests/conftest.py`: the real model class must refuse to be built.
- Per-command refusal and run tests (`tests/test_run.py` and the plugin's refusal tests).
- The plugin's `README.md`, and a line under `## [Unreleased]` in its `CHANGELOG.md`.
- The Commands cell of the plugin's row in the Plugins table of the root `README.md`.
- `NOTICE`, when the module copies a block of the mflux command's `main()`.
