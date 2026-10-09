# mflux-myplugin (template)

This folder is the starting point for a new mflux plugin. It is a complete, tested plugin that adds
`mflux-generate-z-image-myplugin`: mflux's `mflux-generate-z-image` with one example option and hooks where
your library attaches. It is never published. After renaming, rewrite this README for your plugin.

To start a plugin, copy this folder and rename `myplugin` everywhere, then replace the example option, the
refusal and the `apply` hook with your plugin's own. The repository's [CONTRIBUTING.md](../CONTRIBUTING.md)
describes the rules, and [docs/writing-a-plugin.md](../docs/writing-a-plugin.md) explains how a plugin works.

## Start a plugin

1. Copy `template/` to a new folder at the repository root. The folder name is your plugin's name: lowercase
   letters and digits only, starting with a letter, with no `-` or `_` (PEP 8 discourages underscores in package
   names). The same name is used everywhere else: folder `foobar` gives the distribution `mflux-foobar`, the module
   `mflux.extras.foobar`, the command `mflux-generate-z-image-foobar`, the tags `foobar-v*` and the environment
   `pypi-foobar`. `myplugin` is only a placeholder.
2. Rename the package folder `src/mflux/extras/myplugin/` to `src/mflux/extras/<folder>/`. Then replace `myplugin`
   with your folder name in every file: the code, `pyproject.toml`, `NOTICE`, the tests, this README and
   `AGENTS.md`.
3. In `pyproject.toml`, delete the `Private :: Do Not Upload` classifier, rewrite `description`, set
   `version = "0.1.0"`, put your names in `authors` and `maintainers`, and point the `[project.urls]` links at your
   folder (`tree/main/<folder>`). Then run `uv lock` in the folder.
4. Delete the test `test_the_template_wheel_can_never_be_uploaded` from `tests/test_wheel.py`.
5. Copy `.github/workflows/template.yml` to `.github/workflows/<folder>.yml`. In it, rename `template` to your
   folder name, replace the comment at the top, and make the schedule daily, like `teacache.yml`.
6. Copy `.github/workflows/teacache-release.yml` to `.github/workflows/<folder>-release.yml` and replace `teacache`
   with your folder name everywhere in it. Then remove every mention of mlx-teacache, which the rename turned into
   `mlx-<folder>` or `mlx_<folder>`: the package in `uv pip show`, the `-W` filter for its warning and the comments
   about it. Before the first `<folder>-v*` tag, an org admin sets up the release; the steps are under "For maintainers"
   in [CONTRIBUTING.md](../CONTRIBUTING.md#for-maintainers-a-new-plugin-folder).
7. Add a line `/<folder>/ @<your-handle>` to `.github/CODEOWNERS` and a row to the Plugins table in the root
   `README.md`.
8. Rewrite this README for your plugin, the docstring of `src/mflux/extras/<folder>/__init__.py` (`help()` shows
   it), and in `AGENTS.md` the paragraphs above "Namespace contract" and the CI section.
9. From the repository root, run the checker:

   ```bash
   uv run --python 3.13 --no-project scripts/check_plugins.py
   ```

## What the template shows

- `src/mflux/extras/myplugin/_core.py`: the command flow. It parses with mflux's parser, refuses before any
  weights load and loads with mflux's own step. Then it calls `apply(model, args)`, which attaches your
  library to the loaded model and returns its handle (the template returns `None`). For each seed it calls
  `before_generate(model, args, handle, seed)`, generates with mflux's step, calls
  `after_generate(image, args, handle, state)` with the state `before_generate` returned, and saves the image
  the way the mflux command does. `after_generate` is where a plugin adds its own record to the image's
  metadata.
- `src/mflux/extras/myplugin/z_image.py`: the adapter for one mflux command. It names `ZImageCommand`'s steps,
  adds the example option and refusal, and copies a block of statements from mflux's `main()`, which `NOTICE` and
  the drift test cover.
- `tests/`: tests that never download weights, a namespace test, a wheel test, a capabilities test, a test that the
  plugin leaves MLX's memory limits alone, and a drift test that fails when mflux changes the code the plugin calls
  or copies.

## Development

From this folder:

```bash
uv sync --locked
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run ty check src
```

## License

Apache-2.0. `NOTICE` lists the code this template copies from mflux, which is MIT-licensed.
