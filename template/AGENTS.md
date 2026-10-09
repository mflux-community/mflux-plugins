# AGENTS.md

This is the template plugin. Keep it generic: no library code, no release workflow.

Instructions for coding agents working on `mflux-myplugin`, the `template/` folder of mflux-plugins. `myplugin` is a
placeholder for a new plugin's name, which is also its folder name. After you copy the folder and rename `myplugin`,
these rules apply to your plugin; the README covers what the package does. Rules
for the whole repository are in the root `AGENTS.md`. Run every command below from this folder, except the checker,
which runs from the repository root.

## Namespace contract (do not break)

This package follows the namespace model of [mflux PR #776](https://github.com/mflux-community/mflux/pull/776).

- All code goes in `src/mflux/extras/myplugin/`.
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
| Lint and format check | `uv run ruff check . && uv run ruff format --check .` |
| Type check | `uv run ty check src` |
| Build | `uv build` |
| Repository checks (from the repository root) | `uv run --python 3.13 --no-project scripts/check_plugins.py` |

Run the tests, ruff, ty and the checker before you call a change done.

## Code layout

```text
src/mflux/extras/myplugin/
  __init__.py   # __version__ only
  _core.py      # Adapter, run(): the flow every command shares
  z_image.py    # the Z-Image command: parser, the copied pre-load step, ADAPTER, main()
tests/
  _fakes.py         # FakeZImage and the fake plugin library
  _fingerprint.py   # AST fingerprints for the drift test
  test_*.py
```

## How a command works

`run(adapter)` reads `sys.argv` through mflux's own parser. Then `refuse` can reject what your plugin can't run,
before any weights load (`parser.error`, exit 2), and `after_checks` prints mflux's own pre-load warnings. Then `run`
calls mflux's `load()`, calls `apply`, and registers mflux's callbacks (MemorySaver last). For each seed it calls
mflux's `generate()` and saves the image. A failure after load prints the traceback and exits 1.

An adapter names mflux's `validate`, `load`, `generate` and `latent_creator` for one command, plus `apply`:

- `apply(model, args)` runs once per run, after `load` and before mflux registers its callbacks. It attaches your
  library to the loaded model and returns the plugin's handle, or `None`. The template returns `None`.

It adds only the optional hooks its command needs:

- `refuse(parser, args, model_config)`;
- `after_checks(args, model_config)`;
- `before_generate(model, args, handle, seed)`, which runs before each image and returns per-seed state (a counter
  read before the run, for example);
- `after_generate(image, args, handle, state)`, which runs after each image and before it is saved. It gets the
  handle from `apply` and the state from `before_generate`. Add your record to a new `generation_parameters` dict;
  don't change the one the image holds.

The Z-Image adapter's `after_checks` hook (`warn_ineffective_options`) copies a block of statements from mflux's
`main()`, because mflux has no `prepare()` step for it yet. Code copied from mflux is listed in `NOTICE`; a new adapter
that copies more adds its file there.

## Compile and memory

- mflux builds Z-Image's prediction step with `ZImage._predict(transformer)` and wraps it in `mx.compile`, with no
  `inputs` or `outputs`, on M1 and M2 Max and Ultra chips and on M3 and newer chips, but not on base or Pro M1 and M2
  chips (`mflux/models/z_image/variants/z_image.py` and `AppleSiliconUtil.is_m1_or_m2` in
  `mflux/utils/apple_silicon.py`). CI runs on a base M1, so it never runs the compiled path. A compiled step runs
  your Python code only while it is traced, not on every step.
- A plugin that keeps any per-step state (Python values or MLX arrays) or makes a per-step decision must replace
  `_predict` on the instance with an eager factory, as mlx-teacache does. Only a pure, stateless array transform may
  run inside mflux's compiled step; `.item()`, `mx.eval` and writes to outside arrays all fail there. On the chips
  that compile, an eager step also changes the speed and, slightly, the output, so measure your plugin against plain
  mflux running the same step eagerly.
- `_predict` is a factory `(transformer) -> predict`. mflux calls it once per image with its transformer
  (`predict = self._predict(self.transformer)` in `ZImage.generate_image`) and deletes the step after the loop. Build
  your step from that `transformer` argument and never close over `model.transformer`, or `--low-ram` keeps the
  weights alive through the decode. mlx-teacache's `make_teacache_predict_factory` is a worked example.
- Don't keep the transformer or step-sized arrays alive in the handle. Holding the model is fine: mflux frees the
  transformer by setting `model.transformer` to `None`. With one seed and `--low-ram` or `--pid-decode`, mflux drops
  the transformer after the denoising loop to free memory before the decode (`mflux/callbacks/callback_manager.py`),
  and a reference the plugin holds defeats that. Release large arrays in an after-loop callback that `apply`
  registers (it runs before mflux's memory saver), in an interrupt callback, and, for callers that run the command
  in-process, in a `finally`, so an exception doesn't leave them alive.
- Never call `mx.set_wired_limit`, `mx.set_memory_limit`, `mx.set_cache_limit` or anything in `mx.metal` in a
  plugin. These limits are process-wide: mflux sets the cache limit from `--low-ram` and `--mlx-cache-limit-gb`, and
  the wired and memory limits belong to whoever runs the command. `tests/test_memory_limits.py` checks `src/`.
- A `call_before_loop` your plugin registers runs before mflux's battery check, so keep it cheap and safe to run
  again.
- Values you add in `after_generate` must be plain JSON values (`float(...)`, `int(...)`, `str`), not MLX or NumPy
  scalars.
- Z-Image passes no `denoised` to in-loop callbacks, and the latents it passes are not evaluated yet:
  `ZImage.generate_image` calls `ctx.in_loop(t, latents)` before `mx.eval(latents)`. Don't keep them across steps.

## When the drift test fails

`tests/test_mflux_drift.py` pins mflux's Z-Image `build_parser()`, `main()`, `ZImage.generate_image`,
`ZImage._predict` and `AppleSiliconUtil.is_m1_or_m2` by AST digest. It also pins the parser's option strings and
dests, the part of `main()` after `load()` and the step signatures, and it checks that the copied block still matches
`main()`. When an mflux update turns it red:

1. Read mflux's diff of `src/mflux/models/z_image/cli/z_image_generate.py`.
2. Bring `z_image.py` (and `_core.run`, if the flow after `load()` changed) in line with it. A new parser option
   that changes what the plugin must refuse needs a refusal and an entry in the refusal tests.
3. If `ZImage.generate_image` changed, bring `FakeZImage.generate_image` in `tests/_fakes.py` in line with it.
4. Record the digest the failure message prints. Never update a digest without step 1.

## Tests

- No test can download a model. `tests/conftest.py` sets `HF_HUB_OFFLINE=1` and points `HF_HOME`, `HF_HUB_CACHE`
  and `HF_XET_CACHE` at an empty temporary directory before anything imports mflux or huggingface_hub, because both
  read these once, at import.
  It also makes the real `ZImage` class refuse to be built and runs every test in its own temporary directory.
  `tests/test_isolation.py` checks all of that.
- Fakes stand in for the mflux model class and for your library only; everything else is real mflux code.
- After renaming, the run tests call your real `apply` on `FakeZImage`. Give your library a fake in `tests/_fakes.py`
  for them to use.
- Each test's docstring starts with `Bug:` and names the one-line bug that turns it red.
- Warnings are errors in the test run. Expect a warning with `pytest.warns`, or assert there is none with
  `warnings.catch_warnings()` plus `warnings.simplefilter("error")`. Never add a blanket ignore.

## Dependencies and version pins

- The only runtime dependency is `mflux>=0.22`. Add your library next to it with its own lower bound.
- A requirement on `mflux` is a lower bound only. Never give it an upper bound here, because a newer mflux runs and
  warns. `tests/test_wheel.py` checks this.
- Dependencies come from PyPI. Don't add a `sources` table under `[tool.uv]`, or any git or path source, to a committed
  file.
- `ruff` and `ty` are pinned exactly. After you change `pyproject.toml`, run `uv lock` and commit `uv.lock`.

## CI

`.github/workflows/template.yml` (at the repository root) runs on pull requests and pushes to `main` that touch
`template/` or the workflow itself. It also runs all its jobs once a week. There is no release workflow, because the
template is never published. A plugin's release workflow calls its copy of this workflow (`workflow_call`, with
`blocking-only: true`) to run only the `test` job before it publishes. The root `lint.yml` runs on every pull request.

- The `test` job runs ruff, ty and pytest, including the built-wheel test, in the locked environment on Python 3.10
  and 3.13.
- The `mflux-main` job installs mflux from its `main` branch and may fail on pull requests and pushes without
  blocking them. A red drift step means mflux changed something this package copies or pins: follow "When the drift
  test fails".

## Pull requests

- One topic per pull request. Tests, ruff, ty and the checker pass.
- Update `README.md` when a command, option or refusal changes.
- Don't commit personal paths, models, generated images, `.venv*/` or `dist/`.
