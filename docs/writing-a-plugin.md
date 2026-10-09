# Writing a plugin for mflux

This guide is for Python developers who use mflux and want to add a command to it as a plugin. It covers the
whole path: check that your idea fits, propose it, build the plugin from `template/`, test it, get it reviewed and
release it. The process rules (the proposal, pull requests, the admin setup) are in
[CONTRIBUTING.md](../CONTRIBUTING.md). This page covers the code.

## What a plugin is

A plugin is a small Python package that installs next to mflux and adds a command. The command accepts every option
of an existing mflux command and runs that command's own steps to load the model and generate. In between, it
attaches a library to the loaded model.

Command steps are the `validate`, `load` and `generate` methods of an mflux command's `<X>Command` class, plus its
`latent_creator`. Only some mflux commands have them so far.

For example, `mflux-teacache` adds `mflux-generate-z-image-teacache`. It reuses `mflux-generate-z-image` and attaches
TeaCache step skipping from the mlx-teacache library.

A plugin is thin glue:

- The algorithm lives in a separate library, with its own tests and its own releases on PyPI. The plugin only connects
  that library to mflux's command line.
- The plugin does not change mflux or mflux's own commands. It adds a new command next to them.
- It is not a model port. A new model is proposed to mflux itself.

The plugin's code lives under `mflux.extras.<name>`, a namespace mflux leaves open for add-ons.[^namespace] The package
ships only `mflux/extras/<name>/`. It never ships `mflux/__init__.py`, which belongs to mflux, or
`mflux/extras/__init__.py`, which would hide every other plugin. Each plugin is a folder in this repository with
its own lock file, tests, version and releases.

## Is my idea a plugin?

Check three things before you write code:

1. The mflux command you want to extend has command steps. The list is under
   [Which mflux commands can host a plugin](#which-mflux-commands-can-host-a-plugin).
2. The algorithm lives in a library on PyPI that has its own tests and releases.
3. The plugin does not repeat a feature mflux already has. If the plugin can't run together with one, it refuses that
   combination before any weights load.

Then open an issue with the ["Propose a plugin"][propose] form and wait for a maintainer's yes before you open a
pull request.
The full acceptance criteria are in [CONTRIBUTING.md](../CONTRIBUTING.md).

## Start from the template

`template/` is a complete, tested plugin called `mflux-myplugin`. It adds `mflux-generate-z-image-myplugin`, which is
`mflux-generate-z-image` plus one example option and empty hooks where your library attaches. It is never published.

Follow the steps in [template/README.md](../template/README.md) in order. Don't skip step 3 (delete the
`Private :: Do Not Upload` classifier, then run `uv lock`) or step 4 (delete
`test_the_template_wheel_can_never_be_uploaded`). With the classifier left in, the checker fails and PyPI refuses
the upload. With that test left in, the wheel test goes red once the classifier is gone. Step 5 copies
`template.yml` and step 6 copies `teacache-release.yml`.

Then run the checker from the repository root:

```bash
uv run --python 3.13 --no-project scripts/check_plugins.py
```

It prints one line per problem, or `check_plugins: N plugin folder(s) follow the rules`. It checks the folder's
layout, names, license, mflux range and workflow wiring, and it finds a `myplugin` the rename missed. The same check
runs on every pull request.

For a fuller example, read [teacache/](../teacache/). It shows refusals, extra warnings and the record a plugin adds
to each image's metadata.

## Names

The folder name is the plugin's name, and every other name follows from it. It uses lowercase letters and digits
only, starts with a letter and is not a Python keyword.[^folder]

| What | Rule | Example for `foobar` |
|---|---|---|
| Folder | `<name>` | `foobar/` |
| PyPI distribution | `mflux-<name>` | `mflux-foobar` |
| Python module | `mflux.extras.<name>` | `mflux.extras.foobar` |
| Command | the mflux command, then `-<name>` | `mflux-generate-z-image-foobar` |
| Release tag | `<name>-v<version>` | `foobar-v0.1.0` |
| GitHub environment for PyPI | `pypi-<name>` | `pypi-foobar` |
| Workflows | `<name>.yml` and `<name>-release.yml` | `.github/workflows/foobar.yml` |
| CODEOWNERS line | `/<name>/ @<handle>` | `/foobar/ @your-handle` |

Before you pick a name, check the Plugins table in the root [README.md](../README.md) and PyPI for `mflux-<name>`.

The command name matters in two places:

- `mflux-capabilities` is mflux's machine-readable list of the installed commands and the options each one honors.
  It lists a command only if its name starts with `mflux-generate`, `mflux-concept` or
  `mflux-upscale`.[^capabilities] A name built from an mflux command keeps that start. The checker checks only that a
  command starts with `mflux-` and ends with `-<name>`; the template's capabilities test catches a missing prefix.
- Shell completions never include plugin commands. mflux's completion installer reads only mflux's own entry points.

## How a command works

Every command runs through `run(adapter)` in the plugin's `_core.py`. The adapter names mflux's steps for one
command. `run` calls them in the order the mflux command's `main()` does, with your hooks in between:

1. Parse the command line with the mflux command's own `build_parser()`, extended with your options.
2. `validate`: mflux checks the request without loading weights. A config error stops the run with exit code 2.
3. `refuse` (your hook): reject anything the plugin can't run with `parser.error`, which exits with code 2. No
   weights have loaded yet.
4. `after_checks` (your hook): print the pre-load warnings that mflux's `main()` prints.
5. `load`: mflux builds the model.
6. `apply` (required): attach your library to the loaded model. Return a handle for the later hooks, or `None`.
7. mflux registers its callbacks.
8. For each seed: `before_generate` (your hook), mflux's `generate`, `after_generate` (your hook), then save the image
   the way mflux does.

If something fails after the model has loaded, the command prints the traceback and a line that points to the plain
mflux command, then exits with code 1. Ctrl-C during the denoising steps and an unreadable prompt file print a
message and exit with code 0.

This is the template's adapter for Z-Image, from `template/src/mflux/extras/myplugin/z_image.py`:

```python
ADAPTER = Adapter(
    command="mflux-generate-z-image-myplugin",
    plain_command="mflux-generate-z-image",
    build_parser=build_parser,
    validate=ZImageCommand.validate,
    load=ZImageCommand.load,
    generate=ZImageCommand.generate,
    latent_creator=ZImageCommand.latent_creator,
    apply=apply,
    refuse=refuse,
    after_checks=warn_ineffective_options,
)


def main() -> None:
    run(ADAPTER)
```

In the same module, `build_parser()` starts from mflux's parser and adds an argument group for the plugin's options.
The module also re-exports the option constants of the mflux command it extends, so `mflux-capabilities` describes
those options the same way for your command.[^constants] For Z-Image that is `CONDITIONAL_OPTIONS`.

Each mflux command gets its own module. To support a second command, add a module like `z_image.py` with its own
adapter and a second entry in `[project.scripts]`. Plugins share no code with each other: each keeps its own `_core.py`.

The template's fakes, drift pins, capabilities test and conftest guard name Z-Image. For another command, rewrite
them for that command's model class and `main()`.

## Reuse mflux code, don't copy it

Call the mflux command's `build_parser()` and its command steps. Don't write your own version of them, and don't
copy a command's `main()`.

Sometimes `main()` does something no step covers. The template has one case: mflux's Z-Image `main()` warns about
options that have no effect, and no step does that, so the template copies those statements into
`warn_ineffective_options`. When you have to copy:

1. Copy statement for statement, and mark the copy with a comment.
2. List the copied code in `NOTICE` with mflux's MIT license text. Keep `NOTICE` in `license-files` in
   `pyproject.toml`, so the wheel ships it. The checker checks that.
3. Add a drift test that fails when mflux changes the original. The template's `tests/test_mflux_drift.py` does this.
4. Propose the missing step to mflux in its own pull request. Once an mflux release has it, raise your mflux floor
   and delete the copy.

The drift test records a digest of each mflux function the plugin calls or copies.[^drift] When mflux changes one,
the test fails and prints the new digest. Read mflux's diff first and update the plugin to match. Only then record
the new digest. [template/AGENTS.md](../template/AGENTS.md#when-the-drift-test-fails) lists the steps under "When
the drift test fails".

## Which mflux commands can host a plugin

A plugin can extend only an mflux command that has command steps. In mflux 0.22, these commands have them:

| mflux command | Steps class |
|---|---|
| `mflux-generate-z-image` | `ZImageCommand` |
| `mflux-generate-z-image-turbo` | `ZImageTurboCommand` |
| `mflux-generate-z-image-controlnet` | `ZImageTurboControlnetCommand` |
| `mflux-generate-ernie-image` | `ErnieImageCommand` |
| `mflux-generate-ernie-image-turbo` | `ErnieImageTurboCommand` |
| `mflux-generate-krea2` | `Krea2Command` |
| `mflux-generate-lens` | `LensCommand` |
| `mflux-generate-boogu` | `BooguImageCommand` |
| `mflux-generate-ideogram4` | `Ideogram4Command` |
| `mflux-generate-qwen` | `QwenImageCommand` |
| `mflux-generate-qwen-edit` | `QwenImageEditCommand` |
| `mflux-generate-qwen-2.1-controlnet` | `Qwen21ControlnetCommand` |

The other commands have no steps yet. Among them are the FLUX.1 and FLUX.2 commands, FIBO, Ming, Qwen 2.1 generate
and edit, and the upscalers. A model family gets a plugin command only once its mflux command has steps.

This list is a snapshot and goes out of date. To refresh it, run this in an mflux checkout:

```bash
grep -rn "^class .*Command" src/mflux/models/*/cli/
```

Each model's README in mflux shows how to call its steps from Python. The fullest example is the
[Z-Image README](https://github.com/mflux-community/mflux/blob/main/src/mflux/models/z_image/README.md).

## Versions and dependencies

- Require mflux with a lower bound only, for example `mflux>=0.22`. The floor is the first mflux release that has
  every step, parser default and option your command uses.[^floor]
- Never give mflux an upper bound. Users install with `uv tool install mflux --with-executables-from mflux-<name>`,
  and a cap would make the installer pick an older mflux without telling them. The cost: a new mflux release can
  break the plugin. The scheduled `mflux-main` CI job warns you early, and the fix is a plugin release.
- If you need an upper limit, it comes from the library you wrap. teacache gets its limit from mlx-teacache's
  `[mflux]` extra, which lists the mflux versions mlx-teacache has checked.
- Every dependency comes from PyPI: no `[tool.uv.sources]` table, no git or path requirements.
- `ruff` and `ty` (the type checker) are pinned exactly in the `dev` group, and the `ruff==` pin equals the ruff
  version in `.github/workflows/lint.yml`.
- After you change `pyproject.toml`, run `uv lock` and commit `uv.lock`. With an out-of-date lock, `uv sync --locked`
  fails, and so does the `test` job.
- The license is Apache-2.0, and the folder's `LICENSE` is an exact copy of the root one.

The checker enforces the mflux floor and the missing cap, the PyPI-only rule and the ruff pin. CI compares the
`LICENSE` copies.

## Tests

Run these from the plugin's folder:

```bash
uv sync --locked
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run ty check src
```

The template's tests carry over to your plugin with the rename. They follow these rules:

- No test downloads weights. `tests/conftest.py` takes the Hugging Face Hub offline and points its caches at an empty
  temporary folder before anything imports mflux. It also stops the real model class from being built.
  `tests/test_isolation.py` checks all of that.
- Fakes stand in for mflux's model class and for your library only. Everything else is real mflux code. Give your
  library a fake in `tests/_fakes.py`.
- Warnings are errors (`filterwarnings = ["error"]` in `pyproject.toml`). Expect a warning with `pytest.warns`. Never
  add a blanket ignore.
- Each test's docstring starts with `Bug:` and names the one-line bug that would turn the test red.

Each test file guards one part of the contract: the namespace (`test_namespace.py`), the built wheel
(`test_wheel.py`), the `mflux-capabilities` record (`test_capabilities.py`), MLX memory limits
(`test_memory_limits.py`), the command flow and exit codes (`test_run.py`) and the mflux code the plugin uses
(`test_mflux_drift.py`).

## Compile and memory

Read the "Compile and memory" section of [template/AGENTS.md](../template/AGENTS.md#compile-and-memory) before your
`apply` attaches anything to the model. Three of its rules:

- On some chips, mflux compiles the prediction step with `mx.compile`. Your Python code inside a compiled step runs
  only while the step is traced, not on every step. A plugin that keeps per-step state replaces the step with one
  that is not compiled.
- Never set MLX memory limits (`mx.set_wired_limit`, `mx.set_memory_limit`, `mx.set_cache_limit` or anything in
  `mx.metal`). They apply to the whole process and belong to mflux and to whoever runs the command.
- Don't keep the transformer alive in your handle. mflux can free it to save memory (with `--low-ram`, for example),
  and a reference you hold stops that.

## CI

Three workflows test a published plugin: the shared `lint.yml` and two of the plugin's own.

- `lint.yml` runs on every pull request and every push to `main`. It runs ruff over the whole tree, actionlint, the
  license comparison, the checker and the checker's tests. Its `repo` job is the required check.
- `<name>.yml` runs on pull requests, and on pushes to `main`, that touch the folder or the workflow. It also runs
  once a day on a schedule (the template: once a week) and when the release workflow calls it. Its `test` job runs
  ruff, ty and pytest in the locked environment on macOS, with Python 3.10 and 3.13. Its `mflux-main` job runs the
  tests against mflux's `main`.
- `<name>-release.yml` runs on a `<name>-v*` tag. It checks the tag, runs the `test` job, builds, tests the wheel and
  publishes. See [Releases](#releases).

A red `test` job means the plugin is broken against the mflux version in its lock. Fix it before the pull request
merges.[^required]

A red `mflux-main` job means mflux's `main` branch changed something the plugin uses. It does not block a pull
request, but the scheduled run fails for real, so GitHub notifies you. If its drift step is red, read mflux's diff.
A plugin whose library has a public `main` branch can add a `<library>-main` job that works the same way; teacache
has one for mlx-teacache.

Workflow files live in `.github/`, which CODEOWNERS assigns to the repository's maintainers. A change to your
workflows needs their review. The merge rules are under "Pull requests" in [CONTRIBUTING.md](../CONTRIBUTING.md).

## Try a branch before it is released

To install a plugin from a branch, together with mflux, in one tool environment:

```bash
uv tool install mflux --with-executables-from \
  "mflux-<name> @ git+https://github.com/mflux-community/mflux-plugins@<branch>#subdirectory=<name>"
```

You get mflux's commands and the plugin's commands side by side. For a branch in your fork, replace
`mflux-community` with your account. A full commit hash can replace `<branch>`.

To pick up new commits on the branch, run the same command with `--reinstall`:

```bash
uv tool install --reinstall mflux --with-executables-from \
  "mflux-<name> @ git+https://github.com/mflux-community/mflux-plugins@<branch>#subdirectory=<name>"
```

From a checkout, in the plugin's folder:

```bash
uv sync --locked
uv run mflux-generate-z-image-<name> --help
```

Don't install the plugin alone with `uv tool install mflux-<name>`. That builds a second mflux environment and puts
only the plugin's command on your path.

## Releases

Each plugin is released on its own:

1. In a pull request, bump `version` in the plugin's `pyproject.toml` and move the `CHANGELOG.md` entries under the
   new version.
2. After it merges, push the tag `<name>-v<version>` on the merge commit on `main`. The tag ruleset decides who may
   push it.
3. `<name>-release.yml` checks that the tag is `<name>-vX.Y.Z`, that its commit is on `main` and that it equals
   `version`. Then it runs the `test` job, builds the wheel and sdist, and installs the wheel into a fresh environment
   without the lock to run the offline tests again. If any of these fails, the release stops and nothing is
   published.
4. The publish job waits for the required reviewer of the `pypi-<name>` environment. After approval it publishes to
   PyPI through Trusted Publishing, where PyPI accepts the upload from this workflow without a stored token. Then it
   creates the GitHub release.

Never upload to PyPI by hand. Before the first tag, an org admin sets up the environment, the tag ruleset and the
PyPI pending publisher, the entry that lets this workflow create the project on PyPI. Without the pending publisher,
PyPI refuses the upload and the publish job fails. Each plugin has at least two PyPI owners. The checklist is in
[CONTRIBUTING.md](../CONTRIBUTING.md#for-maintainers-a-new-plugin-folder).

## What mflux does and doesn't do for plugins

mflux gives plugins the open `mflux.extras` namespace, the command steps and a listing in `mflux-capabilities`.
Beyond that:

- mflux has no plugin registry and loads no plugin by itself. Installing a plugin changes no mflux command.
- The steps carry no stability promise and can change in any release. Plugins test against the locked mflux and
  mflux `main`, and their maintainers follow the changes.

## Getting listed

- A plugin in this repository gets a row in the Plugins table of the root [README.md](../README.md), in the same pull
  request that adds its folder. The checker fails without it.
- To be mentioned in mflux's own README, open a pull request to mflux that adds the plugin to the "Related projects"
  section.

A package outside this repository can follow the same contract:

- It may put its code under `mflux.extras` with a child name no other package uses. Check the Plugins table and PyPI
  first. Reserving a name on PyPI does not reserve a Python namespace.
- Pick a distribution name that does not read as an official mflux package, for example one that carries your
  project's name instead of a bare `mflux-<feature>`.
- It gets listed the same way, through a pull request to mflux's README.

[propose]: https://github.com/mflux-community/mflux-plugins/issues/new?template=propose-a-plugin.yml

[^namespace]: mflux's `mflux/__init__.py` calls `pkgutil.extend_path`, so other distributions can add children under
    `mflux`. `mflux.extras` has no `__init__.py` anywhere, which makes it an implicit namespace package that several
    plugins share. [mflux PR #776](https://github.com/mflux-community/mflux/pull/776) set up this model for
    `mflux.web`.

[^folder]: The folder name becomes part of a Python module name. PEP 8 discourages underscores in package names, and
    a module named after a keyword, such as `mflux.extras.import`, can't be imported.

[^capabilities]: The prefixes are `COMMAND_PREFIXES` in mflux's
    [`src/mflux/cli/capabilities.py`](https://github.com/mflux-community/mflux/blob/main/src/mflux/cli/capabilities.py).
    It scans the console scripts of every installed package, imports each matching command's module and calls its
    `build_parser()`.

[^constants]: `mflux-capabilities` reads `IGNORED_OPTIONS`, `CONDITIONAL_OPTIONS` and `REJECTED_OPTIONS` from the
    command's module. Without them it reports every option as honored, including those mflux ignores for that model.

[^drift]: The digests are of the functions' syntax trees, so changes to formatting, comments or docstrings in mflux
    don't trip them. The test also pins the parser's option names and the signatures of the steps.

[^floor]: The template starts at `mflux>=0.22`. teacache uses the same floor, because 0.22 is the first mflux whose
    Z-Image parser sets the scheduler default itself.

[^required]: The `test` job is not a required check in the branch rules. A workflow that runs only when its folder
    changes never reports on other pull requests, so only the unfiltered `lint / repo` check can be required.
