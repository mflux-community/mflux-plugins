# Changelog

All notable changes to mflux-teacache are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-31

### Added

- The command `mflux-generate-qwen-teacache`. It takes every option of `mflux-generate-qwen`, runs mflux's own load
  and generate steps, and turns on TeaCache step skipping for Qwen-Image-2512, with a default threshold of 0.25.
  Both classifier-free guidance runs of a step follow one skip decision.
- The command refuses `--model Qwen/Qwen-Image` (any letter case) and the mflux-community copies of the original
  Qwen-Image before any weights load, with exit code 2 and a pointer to running the original checkpoint with
  mlx-teacache from Python.
- A `--model` path or repo other than `Qwen/Qwen-Image-2512` and its mflux-community copies runs with one warning that
  names `Qwen/Qwen-Image-2512`.
- The README reports what the Qwen-Image command did on real weights (mflux 0.24.0, 512×512, `-q 4`, 32 GB M1 Max):
  skip counts, SSIM against plain mflux, and a peak of about 26 GiB at load. Several seeds in one command do not fit
  in 32 GB at that size; run one seed per command.

### Changed

- The checkpoint warning compares repo ids without regard to case, as the Hub does. This also affects the Z-Image
  command: `--model tongyi-mai/z-image` and other letter-case variants of the calibrated checkpoint and its
  mflux-community copies no longer get the warning.

## [0.1.3]

### Added

- On mflux 0.24 and later, which has the new Z-Image decode options, `--vae-tiling` and `--vae-tile-size` no longer
  tile the decode, and the command prints mflux's own warning that they change only the encode of an input image, as
  `mflux-generate-z-image` does. A run that TeaCache refuses prints only the refusal.

### Changed

- Requires mflux 0.24 or newer (was 0.22) and Python 3.11 or newer (was 3.10). mflux 0.24 is the first release that
  imports with Pillow 13, and it needs Python 3.11 itself. On Python 3.10 the installer keeps mflux-teacache 0.1.2,
  because 0.1.3 requires Python 3.11 or newer. On mflux 0.23, installing or upgrading to this release also upgrades
  mflux to 0.24. To stay on 0.1.2, pin `mflux-teacache==0.1.2`.
- mlx-teacache has not been checked on mflux 0.24 yet. On it, the command prints a warning that the installed mflux is
  newer than the ones mlx-teacache was checked on. The plugin's offline tests (fakes and a tiny random transformer)
  show TeaCache still applying on mflux 0.24; no run with real model weights has confirmed it.

### Removed

- Python 3.10 is no longer supported. On Python 3.10 the installer keeps mflux-teacache 0.1.2.
- The Pillow cap (`pillow<13`). mflux 0.24 needs Pillow 13, so the cap is no longer needed.

## [0.1.2]

### Changed

- Requires mlx-teacache 0.14.0 or newer (was 0.13.2). It lists mflux 0.23 as checked, so the command no longer warns
  on mflux 0.23, and its `[mflux]` extra allows mflux below 0.26.

### Fixed

- A fresh `pip install mflux-teacache` on Python 3.11 or newer picked Pillow 13, and the command then failed at
  `import mflux`: every mflux release up to 0.23.0 imports `PIL._typing`, which Pillow 13.0 removed. The package now
  requires Pillow below 13 until an mflux release carries the fix
  ([mflux PR 890](https://github.com/mflux-community/mflux/pull/890)).

## [0.1.1]

### Added

- Each image also records `teacache_plugin_version`, the mflux-teacache version, next to `teacache_version` (the
  mlx-teacache version).

### Changed

- The mflux-community copies of Z-Image (`mflux-community/z-image-base-mflux-q3`, `-q4`, `-q5`, `-q6`, `-q8` and
  `-bf16`) no longer get the checkpoint warning. They are the same model at other precisions (8-bit and lower, or
  bf16). TeaCache's settings were measured only at 8-bit; on the other precisions, skip counts and image quality are
  unmeasured.
- The checkpoint warning for any other repo now says the repo may not be the checkpoint TeaCache's settings were tuned
  on, that skip counts and image quality on it are unmeasured, and that TeaCache still runs.
- A run with too few steps now gets one sentence: TeaCache needs at least 3 denoising steps, because it always
  computes the first and the last one. The exit code is still 2.
- The README names the setup TeaCache's Z-Image settings were measured on.
- The README example passes `--guidance 4.0`. Without it, mflux 0.22 runs Z-Image at guidance 0.

## [0.1.0]

The first release.

### Added

- The command `mflux-generate-z-image-teacache`. It takes every option of `mflux-generate-z-image`, runs mflux's
  own load and generate steps, and turns on TeaCache step skipping from mlx-teacache in between.
- `--teacache-threshold VALUE` replaces the model's calibrated threshold (0.12 for Z-Image).
- Before any weights load, the command stops with exit code 2 for a model TeaCache doesn't support on this command
  (Z-Image Turbo included), a threshold that is not greater than 0 and at most 1, or too few denoising steps.
- Warnings, while the run goes on, for a checkpoint other than the one the coefficients were calibrated on and for
  `--compute-precision`, which TeaCache hasn't been measured with.
- Each image records `teacache_threshold`, `teacache_skipped_steps`, `teacache_active_steps`, `teacache_variant`
  and `teacache_version` next to mflux's own metadata.

Requires Python 3.10 or newer on Apple Silicon, mflux 0.22 or newer, and mlx-teacache 0.13.2 or newer.
