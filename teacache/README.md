# mflux-teacache

Adds the command `mflux-generate-z-image-teacache` to [mflux](https://github.com/mflux-community/mflux). It takes
every option of `mflux-generate-z-image` and runs mflux's own loading and generation steps, with TeaCache step
skipping from [mlx-teacache](https://github.com/IonDen/mlx-teacache) turned on in between.

TeaCache skips transformer steps when the result would barely change, and reuses an earlier result. The image is
close to the plain command's output but not identical. For a supported model, run `mflux-generate-z-image-teacache`
wherever you would run `mflux-generate-z-image`; the other options stay the same.

mflux-teacache lives in the `teacache/` folder of
[mflux-plugins](https://github.com/mflux-community/mflux-plugins), the plugin repository of the mflux-community
organization. Denis Ineshin maintains it. The step skipping itself is in
[mlx-teacache](https://github.com/IonDen/mlx-teacache), a separate Python library by the same author; this package
connects it to mflux's command line.
[Versions](https://github.com/mflux-community/mflux-plugins/tree/main/teacache#versions) lists the mflux releases it
installs with.

## Install

The recommended setup puts mflux and this package in one tool environment:

```bash
uv tool install mflux --with-executables-from mflux-teacache
```

You get mflux's commands and `mflux-generate-z-image-teacache` side by side, and `mflux-capabilities` lists all of
them. In a virtual environment that already has mflux:

```bash
pip install mflux-teacache
```

`uv tool install mflux-teacache` on its own works too, but it builds a second mflux environment and puts only this
package's command on your path.

## Usage

```bash
mflux-generate-z-image-teacache --prompt "a lighthouse on a cliff at dusk" --steps 50 --guidance 4.0 --seed 42 -q 8
```

Every `mflux-generate-z-image` option works the same way. After each image the command prints a line such as
`TeaCache: skipped <k> of <n> steps (threshold 0.12)`.

`--teacache-threshold VALUE` replaces the model's calibrated threshold. It must be greater than 0 and at most 1.
A higher value skips more steps and moves the image further from a plain mflux run.

mflux also has a `--step-cache-ratio` flag (alias `--teacache-ratio`), on the Qwen-Image 2.1 command. It is a
different feature: a fixed share of steps reused by mflux's own step cache. That step cache is not an option on the
Z-Image command.

`--compute-precision` is accepted with a warning: in mflux 0.22 the only value is `float16`, and TeaCache's skip
decisions and image quality have not been measured with it. It changes only mflux's attention and feed-forward math;
TeaCache's cached results keep the model's own precision.

The command checks your request before it loads any weights. When TeaCache can't run it, the command stops with
exit code 2 and a message that points to the plain mflux command. That happens for:

- a model TeaCache doesn't support on this command, such as `--model z-image-turbo` (the message lists the
  supported models);
- a distilled few-step model, where there is nothing to skip;
- a threshold outside 0 to 1;
- mflux's step cache. This command has no step-cache option; a later command for a model that has one refuses it,
  also when `-C` restores it;
- too few steps. The first and last steps always run. 2 or fewer active steps are refused. With 3, the run is accepted
  but skips nothing, and TeaCache prints a warning saying so. An image-to-image run counts only the steps it actually
  runs, so a high strength in `--image PATH STRENGTH` can leave too few (strength 1.0 leaves none).

A checkpoint other than the one TeaCache was tuned on (a finetune, a third-party repo) runs with a warning, because
the skips may not suit it. The mflux-community copies of `Tongyi-MAI/Z-Image` (`mflux-community/z-image-base-mflux-q3`,
`-q4`, `-q5`, `-q6`, `-q8` and `-bf16`) are the same model at other precisions (8-bit and lower, or bf16). They run
without the checkpoint warning. TeaCache's settings were measured only at 8-bit, so on the other precisions its skip
counts and image quality are unmeasured. The `z-image-turbo-*` copies are Z-Image Turbo, which this command refuses.

If something fails after the model has loaded, the command prints the error's traceback and exits with code 1.

## Supported models

| Command | `--model` | Default threshold |
|---|---|---|
| `mflux-generate-z-image-teacache` | `z-image` (the default) | 0.12 |

Z-Image Turbo is not supported: the command refuses it as an unsupported model. Other model families get a
command once their mflux command has the same load-and-generate steps.

## What TeaCache was tuned on

TeaCache's settings for Z-Image, the 0.12 default threshold included, come from runs of one recipe: the
`Tongyi-MAI/Z-Image` checkpoint at 8-bit, 512×512, 50 steps and guidance 4.0. The example under Usage uses the same
steps and guidance, but mflux's default size of 1024×1024. Leave out `--guidance` and mflux 0.22 runs Z-Image at
guidance 0, with classifier-free guidance off.

The settings were fitted on mflux's float32 hidden stream. mflux 0.22 runs Z-Image in bfloat16 by default. There, the
same recipe skipped 12 of 48 active steps instead of 15, at an SSIM of 0.990 against plain mflux (SSIM scores image
similarity; 1.0 means identical), according to the
[Z-Image page](https://github.com/IonDen/mlx-teacache/blob/main/docs/variants/z-image-base.md) in the mlx-teacache
repository. Add `--float32` to run on the stream the settings were fitted on.

Other guidance values, step counts, sizes, LoRAs and precisions all run. How many steps TeaCache skips there, and
how close the image stays to a plain mflux run, has not been measured. The command doesn't warn about these, apart
from the short step counts described under Usage. It warns about the checkpoint only when it is neither
`Tongyi-MAI/Z-Image` nor one of the copies listed under Usage, and it warns about `--compute-precision`.

## What it changes

The command patches the loaded model in memory. For Z-Image it replaces the model's prediction step with
mlx-teacache's gated version. Nothing on disk changes, and mflux's own commands behave exactly as before. mflux's
shell completions list only mflux's own commands.

Each image records what TeaCache did, next to mflux's usual metadata: `teacache_threshold`,
`teacache_skipped_steps`, `teacache_active_steps`, `teacache_variant`, `teacache_version` (the mlx-teacache
version) and `teacache_plugin_version` (the mflux-teacache version). Replaying a TeaCache image with plain mflux and
`-C` produces the plain run, without TeaCache. Replaying it with this command uses the model's default threshold
unless you pass `--teacache-threshold` again, because the `teacache_*` keys are not read back.

For measured speedups and image comparisons, see the
[mlx-teacache README](https://github.com/IonDen/mlx-teacache#readme).

## Versions

- Python 3.10 or newer, on Apple Silicon.
- mflux 0.22 or newer. The upper limit comes from mlx-teacache: its `[mflux]` extra allows mflux below 0.25. A newer
  mflux needs a newer mlx-teacache first; until then the installer keeps or picks an older mflux. mflux 0.21 and
  earlier set the Z-Image scheduler default outside the parser this command reuses, so the command would not get it.
- mlx-teacache 0.13.2 or newer, the first release that lists mflux 0.22 as checked. This package adds no upper limit
  of its own. On an mflux newer than the newest one mlx-teacache was checked against, the command runs and prints a
  warning.

mflux's other options, `--float32` included, pass through unchanged.

What changed in each release is in
[CHANGELOG.md](https://github.com/mflux-community/mflux-plugins/blob/main/teacache/CHANGELOG.md).

## Development

From the `teacache/` folder of an mflux-plugins checkout:

```bash
uv sync --locked
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run ty check src
```

The tests use stand-ins for the model and never download weights. `AGENTS.md` describes the package layout and its
rules.

## License

Apache-2.0. `NOTICE` lists the code this package copies from mflux, which is MIT-licensed.
