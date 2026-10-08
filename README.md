# mflux-plugins

Optional add-ons for [mflux](https://github.com/mflux-community/mflux), the [MLX](https://github.com/ml-explore/mlx) port of FLUX and other image models for Apple Silicon.

Each folder here is its own Python package, with its own tests, version and releases. A plugin installs next to mflux as a separate package, puts its code under mflux's `mflux.extras` namespace and adds its own commands. It does not change mflux or mflux's own commands. Install instructions will be in each plugin's folder once it is published.

## Plugins

| Plugin | Package (planned) | Command (planned) | Status |
|---|---|---|---|
| `teacache/` | `mflux-teacache` | `mflux-generate-z-image-teacache`, more models to follow | In progress, not on PyPI yet |

The `teacache` plugin is built on [mlx-teacache](https://github.com/IonDen/mlx-teacache) and maintained by [@IonDen](https://github.com/IonDen). On some denoising steps, mlx-teacache predicts that the transformer's output would barely change, and reuses what it computed on an earlier step instead of running the transformer's main blocks again. That can make generation faster. How much faster, and how much the image differs from a normal run, depends on the model and settings; the measurements are in the mlx-teacache repository. The plugin will run mflux's own generate command with mlx-teacache applied, and stop with a clear error for a model or setting mlx-teacache does not support.

## Contributing

There is no contributor guide yet. To propose a plugin or report a problem, open an issue in this repository.

## License

Apache-2.0; see [LICENSE](LICENSE). Each plugin folder carries its own copy. A plugin that includes code copied from mflux also has a `NOTICE` file with mflux's MIT license text.
