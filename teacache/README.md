# mflux-teacache

Adds two commands to [mflux](https://github.com/mflux-community/mflux): `mflux-generate-z-image-teacache` and
`mflux-generate-qwen-teacache`. Each takes every option of the mflux command it mirrors (`mflux-generate-z-image` and
`mflux-generate-qwen`) and runs mflux's own loading and generation steps, with TeaCache step skipping from
[mlx-teacache](https://github.com/IonDen/mlx-teacache) turned on in between.

TeaCache skips transformer steps when the result would barely change, and reuses an earlier result. The image is
close to the plain command's output but not identical. For a supported model, run the TeaCache command wherever you
would run the plain one; the other options stay the same.

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

You get mflux's commands and the two TeaCache commands side by side, and `mflux-capabilities` lists all of them. In a virtual environment that
already has mflux:

```bash
pip install mflux-teacache
```

`uv tool install mflux-teacache` on its own works too, but it builds a second mflux environment and puts only this
package's commands on your path.

## Usage

This section shows the Z-Image command; most of it applies to both commands. Qwen-Image has its own section below.

```bash
mflux-generate-z-image-teacache --prompt "a lighthouse on a cliff at dusk" --steps 50 --guidance 4.0 --seed 42 -q 8
```

Every `mflux-generate-z-image` option works the same way. After each image the command prints a line such as
`TeaCache: skipped <k> of <n> steps (threshold 0.12)`.

`--teacache-threshold VALUE` replaces the model's calibrated threshold. It must be greater than 0 and at most 1.
A higher value skips more steps and moves the image further from a plain mflux run.

mflux also has a `--step-cache-ratio` flag (alias `--teacache-ratio`), on the Qwen-Image 2.1 command. It is a
different feature: a fixed share of steps reused by mflux's own step cache. That step cache is not an option on the
Z-Image or Qwen-Image commands.

`--compute-precision` is accepted with a warning: in mflux 0.24 the only value is `float16`, and TeaCache's skip
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

## Qwen-Image

```bash
mflux-generate-qwen-teacache --prompt "a red apple on a wooden table" --width 768 --height 768 --steps 50 --guidance 4.0 --seed 42 -q 4
```

`mflux-generate-qwen-teacache` takes every option of `mflux-generate-qwen`, plus `--teacache-threshold`. The refusals,
the exit codes and the skip line are the same as for Z-Image. Its default threshold is 0.25.

The command runs Qwen-Image-2512, the model that `--model qwen-image` loads (the default) and that `qwen`,
`qwen-image-2512` and `qwen-2512` name as well. The example above uses the recipe the 0.25 threshold was tuned on:
768×768, 50 steps, guidance 4.0 and 4-bit weights (`-q 4`). On that recipe, with mflux 0.23.0, mlx-teacache skipped 29
of 48 active steps at an SSIM of 0.969 against plain mflux. The figures come from one prompt and one seed; the
[Qwen-Image-2512 page](https://github.com/IonDen/mlx-teacache/blob/main/docs/variants/qwen-image-2512.md) in the
mlx-teacache repository has the full sweep.

The command was also run on real weights with mflux 0.24.0, on a 32 GB M1 Max: 512×512, 20 steps, mflux's default
guidance (no `--guidance`), `-q 4`, the default threshold. TeaCache skipped 7 of 20 steps, and the image scored an SSIM
of 0.964 against plain `mflux-generate-qwen` with the same flags (SSIM compares two images; 1.0 means identical).
These are the two commands:

```bash
mflux-generate-qwen-teacache --prompt "a red apple on a wooden table" --steps 20 --width 512 --height 512 -q 4 --seed 42
mflux-generate-qwen --prompt "a red apple on a wooden table" --steps 20 --width 512 --height 512 -q 4 --seed 42
```

At 50 steps and guidance 4.0 (same size, `-q 4`, seed 42), it skipped 31 of 50, and the Python API (`QwenImage` plus
`apply_teacache`) gave a pixel-identical image with the same 31 skips. `--low-ram` skipped the same 7 of 20 and
matched the default run (SSIM 1.000). An image-to-image run (`--image <file> 0.6`, 20 steps) judged only the 8 steps
mflux runs and skipped 2 of them. Replaying a `--low-ram` run at the default guidance from its metadata file with `-C`
gave a pixel-identical image; a run with another guidance does not replay exactly (see "What it changes"). Every figure
comes from one run with one prompt and one seed.

Qwen-Image runs each step twice, once for the prompt and once for the negative prompt. TeaCache makes one skip
decision per step, and both runs follow it.

Memory: every single-seed run above peaked at about 26 GiB (25.7 to 26.2), measured as the process's memory
footprint, and it peaks while the model loads. Plain `mflux-generate-qwen` peaks at the same figure. A 32 GB Mac lets
the GPU use about 25 GiB, so macOS swaps during each load (up to about 10 GB of swap) and then recovers. Close other
apps first.

`--low-ram` and `--pid-decode` free the transformer before the image is decoded. That frees their memory for the
decode, but it does not lower the peak: the peak comes at load, before they free anything, and the denoising loop
does not lower it either.

Run one seed per command. `--seed 42 43` needs more memory than a 32 GB Mac has at 512×512, with or without
`--low-ram`: the first seed's decode passed 28 GiB.

### The original Qwen-Image checkpoint

`--model Qwen/Qwen-Image` is refused, in any letter case, before any weights load. The same goes for the
`mflux-community/qwen-image-mflux-q3`, `-q4`, `-q5`, `-q6`, `-q8` and `-bf16` copies of it. mflux would load those
weights on its Qwen-Image-2512 settings, and TeaCache would then use the 2512 numbers on a model they were not fitted
on. The refusal exits with code 2 and names the Python route below.

To run the original checkpoint with TeaCache, use mlx-teacache from Python:

```python
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mlx_teacache import apply_teacache

model = QwenImage(model_config=ModelConfig.from_name("Qwen/Qwen-Image"), quantize=4)
handle = apply_teacache(model)
```

mlx-teacache then picks the original checkpoint's own settings, with a default threshold of 0.30.

### Other checkpoints

`Qwen/Qwen-Image-2512` and the mflux-community copies of it (`mflux-community/qwen-image-2512-mflux-q3`, `-q4`, `-q5`,
`-q6`, `-q8` and `-bf16`) run without a warning. Any other `--model` path or repo, a finetune or a local folder, runs
with one warning that names `Qwen/Qwen-Image-2512`, because the command cannot tell what is in it. A local folder that
holds the original Qwen-Image gets this warning too, not the refusal.

The 0.25 threshold was measured on 4-bit weights loaded at run time (`-q 4`). The pre-quantized copies, and a
`-q` value other than 4, are the same model at another precision, and TeaCache's skip counts and image quality there
are unmeasured.

## Supported models

| Command | `--model` | Default threshold |
|---|---|---|
| `mflux-generate-z-image-teacache` | `z-image` (the default) | 0.12 |
| `mflux-generate-qwen-teacache` | `qwen-image` (the default), `qwen`, `qwen-image-2512`, `qwen-2512` | 0.25 |

Z-Image Turbo is not supported: the command refuses it as an unsupported model. The Qwen-Image command runs
Qwen-Image-2512 and refuses the original Qwen-Image, as described above. Other model families get a command once
their mflux command has the same load-and-generate steps.

## What TeaCache was tuned on

TeaCache's settings for Qwen-Image-2512, the 0.25 default threshold included, come from runs of the recipe shown in
the Qwen-Image section: 4-bit weights, 768×768, 50 steps and guidance 4.0. The Qwen-Image command's own defaults
differ from it: mflux runs 1024×1024, 20 steps and guidance 3.5 unless you pass other values, and without `-q` it
runs the weights in bfloat16. They run too. How many steps TeaCache skips at 1024×1024 or in bfloat16, and how close
the image stays to a plain mflux run, has not been measured. The measured case is 512×512 with `-q 4`, described in
the Qwen-Image section. The command doesn't warn about these settings. It warns about the checkpoint, as described
there, and about `--compute-precision`.

TeaCache's settings for Z-Image, the 0.12 default threshold included, come from runs of one recipe: the
`Tongyi-MAI/Z-Image` checkpoint at 8-bit, 512×512, 50 steps and guidance 4.0. The example under Usage uses the same
steps and guidance, but mflux's default size of 1024×1024. Leave out `--guidance` and mflux 0.24 runs Z-Image at
guidance 0, with classifier-free guidance off.

The settings were fitted on mflux's float32 hidden stream. mflux 0.24 runs Z-Image in bfloat16 by default. On the
bfloat16 stream, a run of the same recipe on mflux's main branch after the 0.21.0 release skipped 12 of 48 active steps
instead of 15, at an SSIM of 0.990 against plain mflux (SSIM scores image similarity; 1.0 means identical), according
to the [Z-Image page](https://github.com/IonDen/mlx-teacache/blob/main/docs/variants/z-image-base.md) in the
mlx-teacache repository. That run was not repeated on mflux 0.24. Add `--float32` to run on the stream the settings were fitted on.

Other guidance values, step counts, sizes, LoRAs and precisions all run. How many steps TeaCache skips there, and
how close the image stays to a plain mflux run, has not been measured. The command doesn't warn about these, apart
from the short step counts described under Usage. It warns about the checkpoint only when it is neither
`Tongyi-MAI/Z-Image` nor one of the copies listed under Usage, and it warns about `--compute-precision`.

## What it changes

The command patches the loaded model in memory. For Z-Image it replaces the model's prediction step with
mlx-teacache's gated version; for Qwen-Image it swaps the transformer for a wrapper that does the gating. Nothing on
disk changes, and mflux's own commands behave exactly as before. mflux's
shell completions list only mflux's own commands.

Each image records what TeaCache did, next to mflux's usual metadata: `teacache_threshold`,
`teacache_skipped_steps`, `teacache_active_steps`, `teacache_variant`, `teacache_version` (the mlx-teacache
version) and `teacache_plugin_version` (the mflux-teacache version). Replaying a TeaCache image with plain mflux and
`-C` produces the plain run, without TeaCache. Replaying it with this command uses the model's default threshold
unless you pass `--teacache-threshold` again, because the `teacache_*` keys are not read back.

One more thing about replay on Qwen-Image: mflux 0.24.0 writes `"guidance": null` into the sidecar of a Qwen-Image
image, so `-C` replays it at guidance 3.5 whatever the original run used.
mflux has fixed this on its main branch ([mflux PR 952](https://github.com/mflux-community/mflux/pull/952)), after the
0.24.0 release; no release carries the fix yet.

For measured speedups and image comparisons, see the
[mlx-teacache README](https://github.com/IonDen/mlx-teacache#readme).

## Versions

- Python 3.11 or newer, on Apple Silicon.
- mflux 0.24 or newer. It is the first mflux that imports with Pillow 13, which a fresh install picks. The upper limit
  comes from mlx-teacache: its `[mflux]` extra allows mflux below 0.26. A newer mflux needs a newer mlx-teacache
  first; until then the installer keeps or picks an older mflux. On Python 3.10 the installer keeps version 0.1.2 of
  this package. On mflux 0.23, installing or upgrading to this release also upgrades mflux to 0.24; to stay on 0.1.2,
  pin `mflux-teacache==0.1.2`.
- mlx-teacache 0.14.0 or newer, which lists mflux 0.22 and 0.23 as checked. It has not been checked on mflux 0.24: there
  either command runs and prints a warning that the installed mflux is newer than the checked ones. The plugin's
  offline tests (fakes and a tiny random transformer, for both models) show TeaCache still applying there; no run
  with real weights has
  confirmed it. This package adds no upper limit of its own.

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
