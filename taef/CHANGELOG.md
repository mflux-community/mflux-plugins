# Changelog

All notable changes to this plugin are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Two commands, `mflux-generate-z-image-turbo-taef` and `mflux-generate-z-image-taef`. Each takes every option of the
  mflux command it mirrors and writes a live preview image while the picture is denoised, decoded by mlx-taef.
- Three options in a "TAEF preview" group: `--taef-preview PATH`, `--taef-every N` (default: every step) and
  `--taef-frames`, which keeps each step's preview as its own file.
- Default preview path `<output name>.preview.png` next to the output image, carrying `_seed_<n>` with several seeds. An
  explicit `--taef-preview` path is one file overwritten seed after seed, unless it contains `{seed}`; with
  `--taef-frames` and several seeds, each seed keeps its own files.
- A line before each seed that prints where its preview goes.
- Refusals before any weights load, with exit code 2: `--taef-every` below 1, `--taef-every` with `--taef-frames`, a
  preview suffix other than `.png`, `.jpg`, `.jpeg` or `.webp`, a brace field other than `{seed}` (`{seed}` takes a
  width of one to three digits, padded with spaces or with zeros when it starts with 0, optionally followed by `d`), a
  preview path that is an existing directory or a symbolic link (with `--taef-frames`, also a directory or symbolic
  link at a frame name), a preview file name over 255 bytes, a preview path that names any seed's `--output` file (with
  `--taef-frames`, the `<preview name>_stepNN.<suffix>` frame names are compared instead) and an unknown model path.
- The unknown-model refusal applies only while mflux's model-path rules equal those of the checked release (mflux
  0.24.0); on other rules, or rules this package can't read, the name is left to mflux. A missing model path restored
  with `-C` names the file it came from and suggests `--model PATH`.
- The package calls no private mflux name.
- Exit codes: 1 when the decoder can't be prepared (checked before the model loads) or the run fails after loading, 0
  for a finished run, Ctrl-C or a prompt-file error.
- A preview that fails during a run prints one warning and stops previews for that seed; the image is still saved.
- Requires Python 3.11 or newer, mflux 0.24 or newer and mlx-taef 0.9.0 or newer.
