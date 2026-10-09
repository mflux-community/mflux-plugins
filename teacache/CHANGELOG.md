# Changelog

All notable changes to mflux-teacache are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
