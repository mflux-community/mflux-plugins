"""A fake at the boundary this package owns nothing behind: the mflux model class (weights, Metal). Every call is
bound against the real signature, so a keyword the real code doesn't take fails the test. Modelled on mflux's
tests/cli/helpers/command_fakes.py. A plugin that wraps a library adds a fake for that library's entry point here,
built the same way (bound against the real signature, recording its calls)."""

import inspect
from typing import Any, ClassVar

import mlx.core as mx
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.compute_precision.compute_precision import ComputePrecision
from mflux.models.common.config.config import Config
from mflux.models.z_image.variants.z_image import ZImage
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class FakeZImage:
    """Stands in for ZImage. Construction is what ZImageCommand.load() does, so an empty
    `instances` means no weights were loaded. generate_image builds its GeneratedImage the way mflux
    0.22's ZImage.generate_image does (guidance forced to 0.0 on a model without guidance, the scheduler
    default resolved, a real Config, the real ImageUtil.to_image with the same keywords), so a save
    writes a real PNG and a real JSON sidecar with the fields mflux writes: steps, guidance, image_path,
    image_strength, pid_*, and float32 / compute_precision in generation_parameters.

    Two fields are simpler than the real ones, safe because no test here reads them back:
    lora_paths / lora_scales are the values passed to __init__ (the real LoRALoader resolves
    names to local files), and quantize is the requested bit width (the real WeightApplier can
    resolve it from the checkpoint's stored quantization)."""

    init_signature = inspect.signature(ZImage.__init__)  # read at import, before conftest blocks the real __init__
    generate_signature = inspect.signature(ZImage.generate_image)
    instances: ClassVar[list["FakeZImage"]] = []
    fail_on_seed: ClassVar[dict[int, BaseException]] = {}

    def __init__(self, **kwargs: Any) -> None:
        bound = self.init_signature.bind(self, **kwargs)
        bound.apply_defaults()
        self.init_kwargs = kwargs
        self.model_config = bound.arguments["model_config"]
        self.float32 = bound.arguments["float32"]
        self.compute_precision = ComputePrecision(bound.arguments["compute_precision"])
        # The real ZImageInitializer sets these from the loaded weights and LoRALoader; see the class docstring.
        self.bits = bound.arguments["quantize"]
        self.lora_paths = bound.arguments["lora_paths"]
        self.lora_scales = bound.arguments["lora_scales"]
        self.callbacks = CallbackRegistry()
        self.tiling_config = None
        self.generate_calls: list[dict[str, Any]] = []
        # The generation_parameters dict each returned image held, kept by identity: the image may share it.
        self.returned_parameters: list[dict[str, Any]] = []
        type(self).instances.append(self)

    def generate_image(self, **kwargs: Any) -> GeneratedImage:
        bound = self.generate_signature.bind(self, **kwargs)
        bound.apply_defaults()
        self.generate_calls.append(kwargs)
        failure = type(self).fail_on_seed.get(kwargs["seed"])
        if failure is not None:
            raise failure
        call = bound.arguments
        # From here on, the steps of mflux 0.22's ZImage.generate_image that decide what the image records:
        # the same guidance and scheduler resolution, a real Config, and the real ImageUtil.to_image with the
        # same keywords. Only the pixels (a blank decode) and generation_time (no tqdm loop) are made up.
        guidance = call["guidance"] if self.model_config.supports_guidance else 0.0
        scheduler = call["scheduler"]
        if scheduler is None:
            scheduler = "flow_match_euler_discrete" if self.model_config.supports_guidance else "linear"
        config = Config(
            width=call["width"],
            height=call["height"],
            guidance=guidance,
            scheduler=scheduler,
            image_path=call["image_path"],
            image_strength=call["image_strength"],
            model_config=self.model_config,
            num_inference_steps=call["num_inference_steps"],
        )
        self.denoise(config=config, seed=call["seed"], prompt=call["prompt"])
        image = ImageUtil.to_image(
            decoded_latents=mx.zeros((1, 3, 16, 16)),
            config=config,
            seed=call["seed"],
            prompt=call["prompt"],
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            image_path=config.image_path,
            image_strength=config.image_strength,
            generation_time=0.01,
            negative_prompt=call["negative_prompt"],
            pid_decode=call["pid_decode"],
            pid_degrade_sigma=call["pid_degrade_sigma"],
            generation_parameters={
                **({"float32": True} if self.float32 else {}),
                **self.compute_precision.generation_parameters(),
            },
        )
        self.returned_parameters.append(image.generation_parameters)
        return image

    def denoise(self, *, config: Config, seed: int, prompt: str) -> None:
        """No denoising loop and no callbacks: the image is made up."""
