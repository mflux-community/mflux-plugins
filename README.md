# mflux-plugins

Optional add-ons for [mflux](https://github.com/mflux-community/mflux), the [MLX](https://github.com/ml-explore/mlx) port of FLUX and other image models for Apple Silicon.

Each folder here is its own Python package, with its own tests, version and releases. A plugin installs next to mflux as a separate package, puts its code under mflux's `mflux.extras` namespace and adds its own commands. It does not change mflux or mflux's own commands. Each plugin's README covers install, usage and the mflux versions it works with.

## Plugins

| Plugin | PyPI package | Module | Commands | Built on | Maintainer | Status |
|---|---|---|---|---|---|---|
| [`teacache/`](teacache/) | `mflux-teacache` | `mflux.extras.teacache` | `mflux-generate-z-image-teacache`, more models to follow | [mlx-teacache](https://github.com/IonDen/mlx-teacache) | [@IonDen](https://github.com/IonDen) | 0.1.0 on PyPI |

The `teacache` plugin is built on [mlx-teacache](https://github.com/IonDen/mlx-teacache) and maintained by [@IonDen](https://github.com/IonDen). On some denoising steps, mlx-teacache predicts that the transformer's output would barely change, and reuses what it computed on an earlier step instead of running the transformer's main blocks again. That can make generation faster. How much faster, and how much the image differs from a normal run, depends on the model and settings; the measurements are in the mlx-teacache repository. The plugin runs mflux's own load and generate steps for each model, with mlx-teacache applied to the loaded model in between, and stops with a clear error for a model or setting mlx-teacache does not support.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how a plugin folder is laid out, pull requests and releases. To propose a plugin or report a problem, open an issue in this repository. To write a plugin, start from [`template/`](template/) and read the guide, [docs/writing-a-plugin.md](docs/writing-a-plugin.md).

## License

Apache-2.0; see [LICENSE](LICENSE). Each plugin folder carries its own copy. A plugin that includes code copied from mflux also has a `NOTICE` file with mflux's MIT license text.
