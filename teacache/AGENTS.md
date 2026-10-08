# AGENTS.md

Instructions for coding agents working on `mflux-teacache`, the `teacache/` folder of mflux-plugins. People can use
it too; the README covers what the package does. Rules for the whole repository are in the root `AGENTS.md`. Run
every command below from this folder.

## Project summary

- Ships one package, `mflux.extras.teacache`, and one command, `mflux-generate-z-image-teacache`
  (`mflux.extras.teacache.z_image:main`).
- mflux core loads and generates; mlx-teacache patches the loaded model. This package only connects them.

## Namespace contract (do not break)

This package follows the namespace model of [mflux PR #776](https://github.com/mflux-community/mflux/pull/776).

- All code goes in `src/mflux/extras/teacache/`.
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
| Tests on Python 3.10 | `UV_PROJECT_ENVIRONMENT=.venv-py310 uv run --python 3.10 pytest` |
| Lint and format check | `uv run ruff check . && uv run ruff format --check .` |
| Type check | `uv run ty check src` |
| Build | `uv build` |

Run the tests, ruff and ty before you call a change done.

## Code layout

```text
src/mflux/extras/teacache/
  __init__.py   # __version__ only
  _core.py      # Adapter, run(): the flow every command shares
  z_image.py    # the Z-Image command: parser, the copied pre-load step, ADAPTER, main()
tests/
  _fakes.py         # FakeZImage, LoopingFakeZImage, FakeTeaCache
  _fingerprint.py   # AST fingerprints for the drift test
  _marks.py         # warning filters shared by test modules
  test_*.py
```

## How a command works

`run(adapter)` reads `sys.argv` through mflux's own parser. Then it refuses, before any weights load
(`parser.error`, exit 2), when it finds:

- a threshold outside (0, 1];
- a model that mlx-teacache's `match_variant` doesn't accept;
- a distilled model;
- a step-cache setting;
- an active window too short for `check_step_window`, zero active steps included.

A supported model on a checkpoint the coefficients were not fitted on warns and runs. So does any non-default
`--compute-precision`. Then `run` calls mflux's `load()`, applies TeaCache and registers mflux's callbacks
(MemorySaver last). For each seed it calls mflux's `generate()`, records the `teacache_*` metadata if mlx-teacache
committed the generation, prints the skip line and saves the image. A failure after load prints the traceback and
exits 1.

An adapter names mflux's `validate`, `load`, `generate` and `latent_creator` for one command. It adds only the
optional hooks its command needs:

- `before_validate(args)`;
- `after_checks(args, model_config)`;
- `active_steps(args, model_config)`, which defaults to mflux's own `Config.init_time_step`;
- `library_checks_checkpoint`, for a family where mlx-teacache already tells custom checkpoints apart.

The Z-Image adapter's one hook (`warn_ineffective_options`) copies a block of statements from mflux's `main()`,
because mflux has no `prepare()` step for it yet. Code copied from mflux is listed in `NOTICE`; a new adapter that
copies more adds its file there.

## When the drift test fails

`tests/test_mflux_drift.py` pins mflux's Z-Image `build_parser()`, `main()` and `ZImage.generate_image` by AST
digest. It also pins the parser's option strings and dests, the part of `main()` after `load()` (one row per adapter)
and the step signatures, and it checks that the copied block still matches `main()`. When an mflux update turns it
red:

1. Read mflux's diff of `src/mflux/models/z_image/cli/z_image_generate.py`.
2. Bring `z_image.py` (and `_core.run`, if the flow after `load()` changed) in line with it. A new parser option
   that skips or caches steps needs a refusal and an entry in the refusal tests.
3. If `ZImage.generate_image` changed, bring `FakeZImage.generate_image` in `tests/_fakes.py` in line with it.
4. Record the digest the failure message prints. Never update a digest without step 1.

## Tests

- No test can download a model. `tests/conftest.py` sets `HF_HUB_OFFLINE=1` and points `HF_HOME` at an empty
  temporary directory before anything imports mflux or huggingface_hub, because both read these once, at import.
  It also forces huggingface_hub's offline flag, makes the real `ZImage` class refuse to be built, and runs every
  test in its own temporary directory. `tests/test_isolation.py` checks all of that.
- Fakes stand in for the mflux model class and for `apply_teacache` only; everything else is real mflux and
  mlx-teacache code. `tests/test_real_teacache.py` runs the real `apply_teacache` once, on `LoopingFakeZImage`,
  which runs mflux's denoising loop over a tiny transformer with random weights it never loads. `FakeTeaCache`
  builds a real handle through mlx-teacache internals that are not public API (`mlx_teacache.handle.VariantPatch`,
  the `TeaCacheHandle` constructor, `TeaCacheStats.record` and `finalize_last_generation`). If an mlx-teacache
  release changes them, fix the fake, not the plugin.
- Each test's docstring starts with `Bug:` and names the one-line bug that turns it red.
- Warnings are errors in the test run. Expect a warning with `pytest.warns`, or assert there is none with
  `warnings.catch_warnings()` plus `warnings.simplefilter("error")`. Never add a blanket ignore.

## Dependencies and version pins

- Runtime dependencies: `mlx-teacache[mflux]>=0.13.2` and `mflux>=0.22`.
- The mflux upper bound comes from mlx-teacache's extra: below 0.25 with mlx-teacache 0.13.2. The direct `mflux`
  requirement is a lower bound only: 0.22 is the first mflux whose Z-Image parser sets the scheduler default itself.
  Never give an mflux requirement an upper bound here, because a newer mflux runs and warns. `tests/test_wheel.py`
  checks this.
- mlx-teacache 0.13.2 is the first release that records mflux 0.22 as checked, and it follows mflux 0.22's
  bfloat16 Z-Image stream. It prints no warning on mflux 0.22, and the weight-free test in
  `tests/test_real_teacache.py` runs with warnings as errors to show that.
- Both come from PyPI. Don't add a `sources` table under `[tool.uv]`, or any git or path source, to a committed
  file.
- `ruff` and `ty` are pinned exactly. After you change `pyproject.toml`, run `uv lock` and
  commit `uv.lock`.

## CI

`.github/workflows/teacache.yml` (at the repository root) runs on pull requests and pushes to `main` that touch
`teacache/` or the workflow itself. It also runs all its jobs once a day. The root `lint.yml` runs on every pull
request.

- The `test` job is blocking in this workflow; branch protection requires only the unfiltered `lint / repo` check,
  because a path-filtered check never reports on pull requests that don't touch `teacache/`. It runs ruff, ty and
  pytest, including the built-wheel test, in the locked environment (the mflux release on PyPI) on Python 3.10 and
  3.13.
- The `mflux-main` and `mlx-teacache-main` jobs may fail on pull requests and pushes without blocking them. On the
  daily scheduled run they fail for real, so GitHub sends its failure notice. Each runs the drift test
  (`tests/test_mflux_drift.py`) as one step and the rest of the suite as a second step, so a moved copy of mflux code
  and a broken command show up separately. Their uv cache has its own key (`cache-suffix: git-main`), so they never
  save over the locked cache that the `test` job restores.
- `mflux-main` installs mflux from its `main` branch into the locked environment (mflux's own dependencies can move
  too). A red drift step means mflux changed something this package copies or pins: follow "When the drift test
  fails". The second step also goes red when mflux main reports a version newer than the newest one mlx-teacache has
  verified: mlx-teacache then issues `TeaCacheUntestedMfluxWarning`, and the warnings-as-errors setting turns it
  into a failure of the real-library test in `tests/test_real_teacache.py`. The fix for that is an mlx-teacache
  release that records the new mflux, not a change here.
- `mlx-teacache-main` reinstalls only mlx-teacache, from its `main` branch; mflux and every other package stay at
  the locked versions unless mlx-teacache main needs others. Red here means an mlx-teacache change would break this
  package once released.

## Releases

- The version lives in `pyproject.toml` (`version`); it is not derived from git. A release bumps it in a pull
  request.
- After that pull request merges, the maintainer pushes the tag `teacache-v<version>` on the merge commit. Tags carry
  the plugin's folder name because every plugin in this repository is released on its own.
- `.github/workflows/teacache-release.yml` refuses a tag that differs from `project.version` in
  `teacache/pyproject.toml` or isn't on `main`. Otherwise it runs the test job and builds the wheel and sdist from
  `teacache/`. The `test-wheel` job then installs that wheel into a fresh environment without the lock, so it gets
  the newest mflux and mlx-teacache the requirements allow (what `pip install mflux-teacache` gets), and runs the
  offline test suite against the installed package. There, mlx-teacache's warning about an mflux newer than it has
  verified stays a warning, as it is for users, instead of failing the run. Only after that does it publish to PyPI through Trusted
  Publishing and create the GitHub release.
- The tag checks catch mistakes; they are not the access control. That is the `pypi-teacache` environment the
  publish job uses: before the first tag, configure it with a required reviewer and the deployment tag rule
  `teacache-v*`, and keep both.
- Never upload to PyPI by hand.

## Pull requests

- One topic per pull request. Tests, ruff and ty pass.
- Update `README.md` when a command, option, refusal or metadata key changes.
- Don't commit personal paths, models, generated images, `.venv*/` or `dist/`.
