# Changelog

All notable changes to mflux-teacache are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
