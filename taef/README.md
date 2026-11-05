# mflux-taef

Adds two commands to [mflux](https://github.com/mflux-community/mflux): `mflux-generate-z-image-turbo-taef` and
`mflux-generate-z-image-taef`. Each takes every option of the mflux command it mirrors
(`mflux-generate-z-image-turbo` and `mflux-generate-z-image`), runs mflux's own loading and generation steps, and
writes a live preview image while the picture is being denoised. Open the file in Preview or Quick Look, and press
Ctrl-C when the composition is wrong instead of waiting for the last step.

The preview is the image as it stands after the current step, decoded by a tiny decoder from
[mlx-taef](https://github.com/IonDen/mlx-taef). Early steps look noisy and it gets clearer from about the middle of
the run. It is not the final image, which mflux still decodes with the model's own VAE and saves the usual way.

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
| `--taef-frames` | off | Keep every step's preview as its own file, `<preview name>_stepNN.<suffix>`, instead of overwriting one file. It writes every step, so it can't be combined with `--taef-every`. |

### Where the files go

By default the preview sits next to the final image: for `--output out.png` it is `out.preview.png`. With several
seeds, mflux adds `_seed_<n>` to the output name, so the previews get it too.

An explicit `--taef-preview PATH` is one live file, and each seed overwrites it in turn, so a viewer you leave open
follows the run. Put `{seed}` in `PATH` to get one file per seed instead. With `--taef-frames` and several seeds,
`_seed_<n>` is added to the end of the name, so each seed keeps its own `..._seed_<n>_stepNN...` files. The line the
command prints before each seed shows the real path for that seed.

### Previews and `--stepwise-image-output-dir`

mflux's `--stepwise-image-output-dir` decodes every step with the model's full VAE and keeps every step plus a
composite image. A preview goes through the tiny decoder into one file. mlx-taef's
[comparison page](https://github.com/IonDen/mlx-taef/blob/main/COMPARISON.md) has the published decode speed and the
similarity to the full VAE for Z-Image; this package adds no timing figure of its own.

## What the command checks

The command checks your request before it loads any weights. When it can't run, it stops with exit code 2 and a message
that names the plain mflux command to use instead. That happens for:

- `--taef-every` below 1;
- `--taef-every` together with `--taef-frames`;
- a preview file that doesn't end in `.png`, `.jpg`, `.jpeg` or `.webp`;
- a `--taef-preview` with a `{...}` field other than `{seed}`;
- a preview path that names the same file as `--output`. The match ignores letter case, because a default macOS volume
  does. Without this check the last preview would take the name, and mflux would save the final image under another one;
- a `--model` that is not one of mflux's names, not a local path that exists and not a Hugging Face repo id
  (`org/model`);
- a model mlx-taef has no decoder for;
- anything mflux's own option checks refuse.

Exit codes:

| Code | When |
|---|---|
| 0 | The run finished, you pressed Ctrl-C (the last preview stays on disk), or the prompt file couldn't be read. |
| 1 | The decoder couldn't be prepared, which is checked before the model loads and prints `could not prepare the TAEF preview decoder`, or the run failed after the model loaded (the traceback is printed). |
| 2 | The request was refused before any weights loaded. |

If a preview fails during a run (a decode error, a folder you can't write to), the command prints one warning on
stderr, stops previews for that seed and finishes the image. The plugin writes nothing into the image's metadata or the
`--make-conf` sidecar, so the final image's metadata is what the plain command writes.

## What it doesn't do

- No fast final decode. The tiny decoder is for previews; the saved image still uses the model's VAE. mflux's own
  `--low-ram` and the row-strip decode remain the options for decode memory.
- No composite image, GIF or video, no terminal or window display, and no record in the image's metadata.
- No prediction previews for Z-Image: mflux doesn't pass the model's prediction to Z-Image callbacks, so the preview
  decodes the latent.
- No FLUX.1, FLUX.2, Qwen-Image or other model families yet, and no Qwen-Image 2.1.
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
