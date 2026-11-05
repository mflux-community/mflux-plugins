# AGENTS.md

Instructions for coding agents working on `mflux-taef`, the `taef/` folder of mflux-plugins. People can use it too; the
README covers what the package does. Rules for the whole repository are in the root `AGENTS.md`. Run every command
below from this folder.

## Project summary

- Ships one package, `mflux.extras.taef`, and two commands: `mflux-generate-z-image-turbo-taef`
  (`mflux.extras.taef.z_image_turbo:main`) and `mflux-generate-z-image-taef` (`mflux.extras.taef.z_image:main`).
- mflux core loads, generates and saves; mlx-taef decodes the in-progress latent into a preview image. This package
  only connects them.

## Namespace contract (do not break)

This package follows the namespace model of [mflux PR #776](https://github.com/mflux-community/mflux/pull/776).

- All code goes in `src/mflux/extras/taef/`.
- Never add `src/mflux/__init__.py` (mflux core owns it) or `src/mflux/extras/__init__.py` (`mflux.extras` must stay
  an implicit namespace package shared with other extras).
- Don't change `module-name` in `[tool.uv.build-backend]`. Its dotted name is what makes the build ship a namespace
  package.
- `tests/test_namespace.py` and `tests/test_wheel.py` check this. They must pass.
- Inside the package, use relative imports (`from ._core import run`). ty can't resolve absolute `mflux.extras.*`
  imports through mflux's path extension.

## Commands

| Task | Command |
|---|---|
| Install the dev environment | `uv sync --locked` |
| Tests | `uv run pytest` |
| Tests on Python 3.11 | `UV_PROJECT_ENVIRONMENT=.venv-py311 uv run --python 3.11 pytest` |
| Lint and format check | `uv run ruff check . && uv run ruff format --check .` |
| Type check | `uv run ty check src` |
| Build | `uv build` |
| Repository checks (from the repository root) | `uv run --python 3.13 --no-project scripts/check_plugins.py` |

Every uv command against `.venv-py311` needs `--python 3.11`, `uv sync` and `uv run` alike; without it uv rebuilds that
environment on 3.13 from `.python-version`. CI sets `UV_PYTHON` instead.

Run the tests, ruff, ty and the checker before you call a change done.

## Code layout

```text
src/mflux/extras/taef/
  __init__.py       # __version__ only
  _core.py          # Adapter, run(): the flow every command shares, the preview options and path rules
  z_image_turbo.py  # the Z-Image Turbo command: parser, the copied pre-load statements, ADAPTER, main()
  z_image.py        # the Z-Image command: parser, the copied pre-load step, ADAPTER, main()
tests/
  _fakes.py         # FakeZImage, LoopingFakeZImage, FakeLivePreview
  _fingerprint.py   # AST fingerprints for the drift test
  test_*.py
```

## How a command works

`run(adapter)` reads `sys.argv` through mflux's own parser plus the "TAEF preview" option group
(`--taef-preview`, `--taef-every`, `--taef-frames`). Then it refuses, before any weights load (`parser.error`,
exit 2), when it finds:

- `--taef-every` below 1;
- `--taef-every` together with `--taef-frames`;
- a preview suffix outside `.png`, `.jpg`, `.jpeg`, `.webp`;
- a `--taef-preview` with a brace field other than `{seed}`, or a `{seed}` field that can't format a seed;
- a preview path that is the same file as that seed's `--output`, compared case-folded because a default APFS volume is
  case-insensitive. Without the check the last preview holds the name and mflux's save, which never overwrites, writes
  the final image as `name_1.png`.

Then it runs mflux's `validate` (`ModelConfigError` becomes exit 2), refuses a `--model` that is no known name, no
existing path and not `org/model` in shape, and asks mlx-taef for the decoder variant (`UnsupportedMfluxModelError`
becomes exit 2; no PR 1 command reaches it, it is a guard). The adapter's `after_checks` hook (`warn_before_load`)
then prints mflux's own pre-load warnings, so a refused run prints one message.

Then it builds `LivePreviewCallback` before the model loads, because the decoder is downloaded and converted on first
use and a failure should cost seconds, not a model load. Any exception there prints
`<command>: error: could not prepare the TAEF preview decoder: ...` and exits 1. The callback gets exactly `variant=`,
`every=`, `save_to=` (a `Path`), `numbered_frames=` and `on_error="disable"`, and never `flux=`.

After that `run` calls mflux's `load()`, registers the preview callback first and mflux's callbacks after it
(MemorySaver last), and for each seed assigns `preview.save_to` a new `Path`, prints the start line, calls mflux's
`generate()` and saves the image the way mflux does. Ctrl-C and prompt-file errors print and exit 0. Any other failure
after load prints the traceback and exits 1. The plugin writes nothing to `generation_parameters` and calls no
`mx.set_*` or `mx.metal` function.

An adapter names mflux's `validate`, `load`, `generate` and `latent_creator` for one command, its own and the plain
command's names, and optionally `after_checks`. A new command is a new adapter module plus a row in `tests/conftest.py`.

The seed suffix is `_seed_{seed}`. On the default path it comes from `args.output`, which mflux already suffixes for
several seeds. An explicit `--taef-preview` path gets it on its stem only with `--taef-frames`, several seeds and no
`{seed}` of its own. Otherwise an explicit path is one file shared by every seed.

Code copied from mflux is listed in `NOTICE`; a new adapter that copies more adds its file there.

## Couplings to mlx-taef

The plugin uses four names from mlx-taef: `LivePreviewCallback` (its keyword names and its `save_to` and `variant`
attributes), `resolve_kernel_from_model_config`, `UnsupportedMfluxModelError`, and the decoder download that the
callback's constructor triggers. mlx-taef pins the first three in `tests/test_plugin_contract.py` in its own
repository, so a change there that would break this package fails there first. The `mlx-taef-main` CI job is the other
half: it installs mlx-taef from its `main` branch and runs this package's tests. `save_to` must always be assigned a
`Path`, never a `str`: the callback derives its numbered-frame names from it, and a `str` breaks them without an error.

## Memory and callbacks

- Never call `mx.set_wired_limit`, `mx.set_memory_limit`, `mx.set_cache_limit` or anything in `mx.metal` in this
  package. These limits are process-wide: mflux sets the cache limit from `--low-ram` and `--mlx-cache-limit-gb`, and
  the wired and memory limits belong to whoever runs the command. `tests/test_memory_limits.py` checks `src/`.
- Z-Image passes no `denoised` to in-loop callbacks, and the latents it passes are not evaluated yet:
  `ZImage.generate_image` calls `ctx.in_loop(t, latents)` before `mx.eval(latents)`. The preview decodes the latent, and
  nothing here may keep it across steps.
- The preview callback is registered before mflux's own callbacks, so it runs ahead of them in every hook.
- This package does not touch the prediction step, so it doesn't need the eager-factory rule that mlx-teacache follows.

## When the drift test fails

`tests/test_mflux_drift.py` pins, for each command, mflux's `build_parser()`, `main()` and its part after `load()`,
`load()` and the class it returns, and the step signatures; the parser's option strings and dests; and the statements
the adapter copies from `main()`. Once for both commands it pins `ZImage.generate_image` (where `in_loop` sits before
`mx.eval`), `ZImage._predict`, the Apple Silicon chip check, and the seams the preview depends on:
`GenerationContext.in_loop`, `CallbackRegistry.register` and `CallbackManager.register_callbacks`. Each pinned value is
recorded under a label that names the mflux source it was reviewed against. This package has one label, `mflux v.0.24.0
released`, with the versions that source reports (`REPORTED_VERSIONS`). A value the label doesn't hold fails, and so
does an installed mflux whose version the label doesn't list. A new mflux option that starts with `--taef-` would
collide at `build_parser()`; the option-string pin catches it on mflux `main`.

When an mflux update turns it red:

1. Read mflux's diff of the command module that moved: `src/mflux/models/z_image/cli/z_image_generate.py` or
   `src/mflux/models/z_image/cli/z_image_turbo_generate.py`, and for a seam, the file it lives in.
2. Bring `z_image.py` or `z_image_turbo.py` (and `_core.run`, if the flow after `load()` changed) in line with it. A
   new parser option that changes what the plugin must refuse needs a refusal and an entry in the refusal tests. Code
   that uses something only the newer mflux has must still import and run on the oldest supported release.
3. If `ZImage.generate_image` changed, bring `FakeZImage.generate_image` in `tests/_fakes.py` in line with it.
4. Record the values the failure messages print under a label naming the mflux source you reviewed: the release
   (`mflux v.0.24.0 released`), or the `main` commit and the review date (`mflux main <commit>, reviewed <date>`), with
   the versions that source reports. A release keeps its label while the package supports it. Never record a value
   without step 1.

## Tests

- No test can download a model or read the user's decoder cache. `tests/conftest.py` sets `HF_HUB_OFFLINE=1` and
  points `HF_HOME`, `HF_HUB_CACHE` and `HF_XET_CACHE` at an empty temporary directory before anything imports mflux or
  huggingface_hub, because both read these once, at import. For each test it also points
  `mlx_taef.download.CACHE_ROOT` at that test's temporary directory (mlx-taef keeps converted decoders in
  `~/.cache/mlx-taef`, outside `HF_HOME`) and makes `mlx_taef.download.get_or_convert` raise.
- The same file makes the real model class refuse to be built and runs every test in its own temporary directory. A
  command's model class is listed in `REAL_PIPELINES` and its mflux command module in `FAKES` (the name the module
  loads the class by, and the fake that replaces it). `isolated_cli` stops every listed class from being built and
  clears each fake's own `instances` and `fail_on_seed`; `run_command` swaps every fake in and runs the adapter under
  its own command name. Both Z-Image commands load the same class, so they share one row's fake. A new command adds
  its rows there. `tests/test_isolation.py` checks all of that, including that every console script's model class and
  command module are listed.
- Fakes stand in for the mflux model class (`FakeZImage`, `LoopingFakeZImage`) and for `LivePreviewCallback`
  (`FakeLivePreview`) only; everything else is real mflux code. `tests/test_real_taef.py` runs the real
  `LivePreviewCallback` with a decoder of random weights it builds itself, through mflux's real callback dispatcher on
  a fake model that runs mflux's denoising loop, so it covers the seams without a download.
- Each test's docstring starts with `Bug:` and names the one-line bug that turns it red.
- Warnings are errors in the test run. Expect a warning with `pytest.warns`, or assert there is none with
  `warnings.catch_warnings()` plus `warnings.simplefilter("error")`. Never add a blanket ignore.

## Dependencies and version pins

- Runtime dependencies: `mlx-taef[mflux]>=0.9.0` and `mflux>=0.24`.
- mlx-taef 0.9.0 is the first release whose `[mflux]` extra admits mflux 0.24; 0.8.4 caps Pillow below 13, which mflux
  0.24 needs. mflux 0.24 is the first mflux that imports with Pillow 13 ([mflux PR 890](https://github.com/mflux-community/mflux/pull/890)).
- A requirement on `mflux` is a lower bound only. Never give it an upper bound here, because a newer mflux runs and
  warns. `tests/test_wheel.py` checks this, and that the mlx-taef floor admits 0.9.0 and refuses 0.8.4.
- Dependencies come from PyPI. Don't add a `sources` table under `[tool.uv]`, or any git or path source, to a committed
  file.
- `ruff` and `ty` are pinned exactly. After you change `pyproject.toml`, run `uv lock` and commit `uv.lock`.

## CI

`.github/workflows/taef.yml` (at the repository root) runs on pull requests and pushes to `main` that touch `taef/` or
the workflow itself. It also runs all its jobs once a day. The root `lint.yml` runs on every pull request.

- The `test` job runs ruff, ty and pytest, including the built-wheel test, in the locked environment on Python 3.11 and
  3.13. It is blocking in this workflow; branch protection requires only the unfiltered `lint / repo` check, because a
  path-filtered check never reports on pull requests that don't touch `taef/`.
- The `mflux-main` job installs mflux from its `main` branch into the locked environment and may fail on pull requests
  and pushes without blocking them. A red drift step means mflux changed something this package copies or pins: follow
  "When the drift test fails".
- The `mlx-taef-main` job reinstalls only mlx-taef, from its `main` branch. Red here means an mlx-taef change would
  break this package once released.

## Releases

- The version lives in `pyproject.toml` (`version`); it is not derived from git. A release bumps it in a pull request.
- After that pull request merges, the maintainer pushes the tag `taef-v<version>` on the merge commit.
- `.github/workflows/taef-release.yml` refuses a tag that differs from `project.version` in `taef/pyproject.toml` or
  isn't on `main`. Otherwise it runs the test job, builds the wheel and sdist from `taef/`, installs the wheel into a
  fresh environment without the lock and runs the offline suite against it, and only then publishes to PyPI through
  Trusted Publishing and creates the GitHub release.
- The tag checks catch mistakes; they are not the access control. That is the `pypi-taef` environment the publish job
  uses: before the first tag, configure it with a required reviewer and the deployment tag rule `taef-v*`, and keep
  both.
- Never upload to PyPI by hand.

## Pull requests

- One topic per pull request. Tests, ruff and ty pass.
- Update `README.md` when a command, option or refusal changes.
- Don't commit personal paths, models, generated images, `.venv*/` or `dist/`.
