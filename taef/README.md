# mflux-taef

Adds two commands to [mflux](https://github.com/mflux-community/mflux): `mflux-generate-z-image-turbo-taef` and
`mflux-generate-z-image-taef`. Each takes every option of the mflux command it mirrors
(`mflux-generate-z-image-turbo` and `mflux-generate-z-image`), runs mflux's own loading and generation steps, and
writes a live preview image while the picture is being denoised. Open the file in Preview or Quick Look, and press
Ctrl-C when the composition is wrong instead of waiting for the last step.

The preview is the image as it stands after the current step, decoded by a tiny decoder from
[mlx-taef](https://github.com/IonDen/mlx-taef). It is not the final image, which mflux still decodes with the model's
own VAE and saves the usual way.

mflux-taef lives in the `taef/` folder of [mflux-plugins](https://github.com/mflux-community/mflux-plugins), the plugin
repository of the mflux-community organization. Denis Ineshin maintains it. The tiny decoder is in mlx-taef, a separate
Python library by the same author; this package connects it to mflux's command line.
[Versions](#versions) lists what it installs with.

## Install

The recommended setup puts mflux and this package in one tool environment:

```bash
uv tool install mflux --with-executables-from mflux-taef
```

You get mflux's commands and the two TAEF commands side by side. In a virtual environment that already has mflux:

```bash
pip install mflux-taef
```

Installing the package brings in mlx-taef. The first run downloads the decoder and converts it for MLX;
later runs reuse it.

## Usage

Use the Turbo command for Turbo; the base command runs both.

```bash
mflux-generate-z-image-turbo-taef --prompt "a lighthouse on a cliff at dusk" --steps 8 --seed 42
```

Every `mflux-generate-z-image-turbo` option works the same way. For Z-Image base:

```bash
mflux-generate-z-image-taef --prompt "a lighthouse on a cliff at dusk" --steps 50 --guidance 4.0 --seed 42
```

Before each image the command prints where its preview goes, for example
`TAEF preview: image.preview.png, every step (zimage decoder)`.

Three options control the preview:

| Option | Default | What it does |
|---|---|---|
| `--taef-preview PATH` | `<output name>.preview.png`, next to the output image | Where the preview goes: a `.png`, `.jpg`, `.jpeg` or `.webp` file (any letter case). `PATH` may contain `{seed}`. |
| `--taef-every N` | every step | Update the preview every `N` steps; `N` is 1 or more. |
| `--taef-frames` | off | Keep every step as its own file, `<preview name>_stepNN.<suffix>`. |

`--taef-frames` writes every step, so it can't be combined with `--taef-every`. `NN` counts the steps this run
executed, so an image-to-image run, which starts partway through the schedule, numbers from its own first step.

### Where the files go

By default the preview sits next to the final image: for `--output out.png` it is `out.preview.png`. With several
seeds, mflux adds `_seed_<n>` to the output name, so the previews get it too.

An explicit `--taef-preview PATH` is one live file, and each seed overwrites it in turn, so reopen the file to see the
latest preview. Put `{seed}` in `PATH` to get one file per seed instead. With `--taef-frames` and several seeds,
`_seed_<n>` is added to the end of the name, so each seed keeps its own `..._seed_<n>_stepNN...` files. The line the
command prints before each seed shows the real path for that seed.

### Previews and `--stepwise-image-output-dir`

mflux's `--stepwise-image-output-dir` decodes every step with the model's full VAE and keeps every step plus a
composite image. A preview goes through the tiny decoder into one file. mlx-taef's
[comparison page](https://github.com/IonDen/mlx-taef/blob/main/COMPARISON.md) has the published decode speed and the
similarity to the full VAE for Z-Image; this package adds no timing figure of its own.

The two options can run in the same command: mflux writes its step images to its directory and this package writes the
preview file or frames to the preview path.

## Checked on real weights

On 2026-11-09 the TAEF commands below ran on real weights, and the single-seed 512×512 ones were compared with the
plain mflux command with the same options. The machine was an Apple M1 Max with 32 GB, macOS 27.2, Python 3.13,
mflux 0.24.0 and mlx-taef 0.9.0. The prompt was "a red apple on a wooden table" and the seed 42. The models were
`mflux-community/z-image-turbo-mflux-q4` and `mflux-community/z-image-base-mflux-q4`. The first two lines are the plain
commands the TAEF images were compared with; the rest are the TAEF commands:

```bash
mflux-generate-z-image-turbo --model mflux-community/z-image-turbo-mflux-q4 --prompt "a red apple on a wooden table" --steps 8 --seed 42 --width 512 --height 512 --output image.png
mflux-generate-z-image --model mflux-community/z-image-base-mflux-q4 --base-model z-image --prompt "a red apple on a wooden table" --steps 20 --guidance 4.0 --low-ram --seed 42 --width 512 --height 512 --output image.png
mflux-generate-z-image-turbo-taef --model mflux-community/z-image-turbo-mflux-q4 --prompt "a red apple on a wooden table" --steps 8 --seed 42 --width 512 --height 512 --output image.png
mflux-generate-z-image-turbo-taef --model mflux-community/z-image-turbo-mflux-q4 --prompt "a red apple on a wooden table" --steps 8 --seed 42 --width 512 --height 512 --output image.png --taef-frames
mflux-generate-z-image-turbo-taef --model mflux-community/z-image-turbo-mflux-q4 --prompt "a red apple on a wooden table" --steps 8 --seed 42 43 --width 512 --height 512 --output image.png
mflux-generate-z-image-taef --model mflux-community/z-image-base-mflux-q4 --base-model z-image --prompt "a red apple on a wooden table" --steps 20 --guidance 4.0 --low-ram --seed 42 --width 512 --height 512 --output image.png --taef-every 5
mflux-generate-z-image-taef --model mflux-community/z-image-base-mflux-q4 --base-model z-image --prompt "a red apple on a wooden table" --steps 8 --guidance 4.0 --seed 42 --width 1024 --height 1024 --output image.png
```

- Turbo, 512×512: the TAEF image was pixel-identical to the plain one, and a 512×512 preview was written. With
  `--taef-frames` the image was identical again, and 8 per-step preview files were written instead of the single
  preview.
- Turbo with `--seed 42 43`: the seed-42 image was identical to the single-seed plain image, and each seed got its own
  preview.
- Base, 512×512, with `--low-ram` and `--taef-every 5`: the image was identical to the plain one, and a preview was
  written.
- Base, 1024×1024: the TAEF command finished and wrote the image and a 1024×1024 preview. There was no plain run of
  that recipe, so there is no comparison.
- Ctrl-C after step 3 of 8 (the Turbo 512×512 TAEF command): mflux printed "Stopping image generation at step", the
  exit code was 0, no image was written, and the preview stayed on disk.

In the single-image 512×512 Turbo runs above, the TAEF command used more memory than the plain one; that is one
comparison on one machine. The TAEF Turbo command was also started at 1024×1024 and did not finish in this test, so nothing here covers
Turbo at that size.

## What the command checks

The command checks your request before it loads any weights. When it can't run, it stops with exit code 2 and a message
that says why. These refusals also tell you to run the plain mflux command to generate without previews:

- `--taef-every` below 1;
- a preview file that doesn't end in `.png`, `.jpg`, `.jpeg` or `.webp`;
- a `--taef-preview` with a `{...}` field other than `{seed}`, or a `{seed}` field with more than a width. `{seed}`
  takes a width of one to three digits (padded with spaces, or with zeros when it starts with 0), optionally followed
  by `d`, so `{seed:04d}` and `{seed:3}` run. A conversion such as `{seed!r}`, a bare `{seed:d}`, any other spec and a
  width of more than three digits (`{seed:0004}` included) are refused.

These refusals only explain the problem:

- `--taef-every` together with `--taef-frames`;
- a `--taef-preview` path that is an existing directory or a symbolic link. The check runs before anything loads. With
  `--taef-frames` the same goes for an existing directory or symbolic link at any `<preview name>_stepNN.<suffix>`
  frame name in the preview's folder;
- a preview file name over 255 bytes, which is the most a file name can have. A `{seed:999}` width can reach that. With
  `--taef-frames` the longest frame name is measured. Without this check the model would load and the first save would
  fail, which turns the preview off;
- a preview path that names the same file as `--output`, or, with several seeds, as any seed's output (for example
  `--seed 2 12 --output a.png --taef-preview 'a_seed_1{seed}.png'`). The match ignores letter case, because a default
  macOS volume does. With `--taef-frames` the preview path itself is never written, so `--output` is compared with the
  `<preview name>_stepNN.<suffix>` frame names instead. Without this check the last preview would take the name, and
  mflux would save the final image under another one;
- a `--model` that is not one of mflux's names, not a local path that exists and not a Hugging Face repo id
  (`org/model`). On an mflux whose model-path rules differ from the checked release, the name is left to mflux. A
  missing path restored with `-C` names the file it came from;
- anything mflux's own option checks refuse, with mflux's own message.

Exit codes:

- 0: the run finished, you pressed Ctrl-C during generation (the last preview stays on disk), or the prompt file
  couldn't be read.
- 1: the decoder couldn't be prepared (checked before the model loads; it prints `could not prepare the TAEF preview
  decoder`), or the run failed after the model loaded (the traceback is printed).
- 2: the request was refused before any weights loaded.

If a preview fails during a run (a decode error, a folder you can't write to), the command prints one warning on
stderr, stops previews for that seed and finishes the image. The plugin writes nothing into the image's metadata or the
`--make-conf` sidecar, so the final image's metadata is what the plain command writes.

## What it doesn't do

- No fast final decode. The tiny decoder is for previews; the saved image still uses the model's VAE. mflux's own
  `--low-ram` stays the option for decode memory. mflux decodes in row strips by default, and `--vae-full-precision`
  switches that off.
- No composite image, GIF or video, no terminal or window display, and no record in the image's metadata.
- No prediction previews for Z-Image: mflux doesn't pass the model's prediction to Z-Image callbacks, so the preview
  decodes the latent.
- No FLUX.1, FLUX.2, Qwen-Image or other model families yet (Qwen-Image 2.1 included).
- No command that combines TAEF with TeaCache (see the next section).

## With mflux-teacache

The two plugins install into one environment, because each adds only its own part of the `mflux.extras` namespace:

```bash
pip install mflux-taef mflux-teacache
```

They don't combine on the command line: `mflux-generate-z-image-teacache` has no `--taef-preview`, and the TAEF
commands don't skip steps. The libraries do work together from Python. TeaCache supports Z-Image base, not Turbo:

```python
from pathlib import Path

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.z_image.variants.z_image import ZImage
from mlx_taef.integrations.mflux import LivePreviewCallback
from mlx_teacache import apply_teacache

model = ZImage(model_config=ModelConfig.z_image(), quantize=8)
handle = apply_teacache(model)
preview = LivePreviewCallback(flux=model, every=1, save_to=Path("preview.png"))
model.callbacks.register(preview)
image = model.generate_image(
    seed=42, prompt="a lighthouse on a cliff at dusk", width=512, height=512, guidance=4.0, num_inference_steps=50
)
image.save("out.png")
```

## Versions

- Python 3.11 or newer, on Apple Silicon.
- mflux 0.24 or newer, with no upper limit.
- mlx-taef 0.9.0 or newer.

What changed in each release is in
[CHANGELOG.md](https://github.com/mflux-community/mflux-plugins/blob/main/taef/CHANGELOG.md).

## Development

From the `taef/` folder of an mflux-plugins checkout:

```bash
uv sync --locked
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run ty check src
```

The tests use stand-ins for the model and a random decoder, and never download weights. `AGENTS.md` describes the
package layout and its rules.

## License

Apache-2.0. `NOTICE` lists the code this package copies from mflux, which is MIT-licensed.
