"""Fakes at the two boundaries this package owns nothing behind: the mflux model class (weights, Metal) and
mlx-taef's LivePreviewCallback (its constructor builds the decoder, which can download). Every call is bound against
the real signature, so a keyword the real code doesn't take fails the test. Modelled on mflux's
tests/cli/helpers/command_fakes.py."""

import inspect
from typing import Any, ClassVar

import mlx.core as mx
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.compute_precision.compute_precision import ComputePrecision
from mflux.models.common.config.config import Config
from mflux.models.common.vae.block_decoder import VAEBlockDecoder
from mflux.models.common.vae.vae_decode_options import VAEDecodeOptions
from mflux.models.z_image.latent_creator import ZImageLatentCreator
from mflux.models.z_image.variants.z_image import ZImage
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil
from mlx_taef.integrations.mflux import LivePreviewCallback


class FakeVAE:
    """Stands in for ZImage.vae, which only the decode reads. It carries the decode mode ZImageInitializer sets (row
    strips) for VAEDecodeOptions.apply to switch."""

    def __init__(self) -> None:
        self.decode_mode = VAEBlockDecoder.LOW_MEMORY


class FakeZImage:
    """Stands in for ZImage. Construction is what ZImageCommand.load() does, so an empty
    `instances` means no weights were loaded. generate_image builds its GeneratedImage the way the locked
    mflux's ZImage.generate_image does (guidance forced to 0.0 on a model without guidance, the scheduler
    default resolved, a real Config, the real ImageUtil.to_image with the same keywords), so a save
    writes a real PNG and a real JSON sidecar with the fields mflux writes: steps, guidance, image_path,
    image_strength, pid_*, and float32 / compute_precision / vae_full_precision (and, on mflux main e9a5fd7,
    small_text_encoder) in generation_parameters.

    Two fields are simpler than the real ones, safe because no test here reads them back:
    lora_paths / lora_scales are the values passed to __init__ (the real LoRALoader resolves
    names to local files), and quantize is the requested bit width (the real WeightApplier can
    resolve it from the checkpoint's stored quantization)."""

    PIPELINE: ClassVar[type] = ZImage
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
        # ZImage.__init__ takes small_text_encoder from mflux main e9a5fd7 on; mflux 0.24.0 has no such keyword.
        self.small_text_encoder = bound.arguments.get("small_text_encoder", False)
        self.compute_precision = ComputePrecision(bound.arguments["compute_precision"])
        # The real ZImageInitializer sets these from the loaded weights and LoRALoader; see the class docstring.
        self.bits = bound.arguments["quantize"]
        self.lora_paths = bound.arguments["lora_paths"]
        self.lora_scales = bound.arguments["lora_scales"]
        self.callbacks = CallbackRegistry()
        self.tiling_config = None
        self.vae = FakeVAE()
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
        # From here on, the steps of the locked mflux's ZImage.generate_image that decide what the image records:
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
                **VAEDecodeOptions.generation_parameters(self),
                # From mflux main e9a5fd7 on, ZImage._generation_parameters() adds this key; 0.24.0 never sets it.
                **({"small_text_encoder": True} if self.small_text_encoder else {}),
            },
        )
        self.returned_parameters.append(image.generation_parameters)
        return image

    def denoise(self, *, config: Config, seed: int, prompt: str) -> None:
        """No denoising loop and no callbacks: the image is made up."""


class _Transformer:
    """Stands in for the transformer the loop holds; plain, so a test can take a weak reference to it."""


class LoopingFakeZImage(FakeZImage):
    """FakeZImage whose denoise runs the locked mflux's loop shape (ZImage.generate_image in
    mflux/models/z_image/variants/z_image.py): the seed's noise latents, the callback context's before_loop, one
    scheduler step per time step on made-up noise, in_loop before mx.eval, the Ctrl-C path through interruption, and
    after_loop. interrupt_at_step raises KeyboardInterrupt at that step, as Ctrl-C would."""

    interrupt_at_step: ClassVar[int | None] = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.transformer: _Transformer | None = _Transformer()
        self.final_latents: list[mx.array] = []

    def denoise(self, *, config: Config, seed: int, prompt: str) -> None:
        latents = ZImageLatentCreator.create_noise(seed, config.height, config.width)
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        for t in config.time_steps:
            try:
                noise = mx.random.normal(latents.shape, key=mx.random.key(t)).astype(latents.dtype)
                latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
                if t == type(self).interrupt_at_step:
                    raise KeyboardInterrupt
                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203 -- the loop shape mflux has
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )
        ctx.after_loop(latents)
        self.final_latents.append(latents)


class FakeLivePreview:
    """Stands in for mlx_taef.integrations.mflux.LivePreviewCallback, whose constructor builds the tiny decoder
    (a download into ~/.cache/mlx-taef on a fresh machine). Construction is bound against the real signature; each
    hook the real callback has records what this seed's preview would write to. fail_on_init makes construction
    fail the way a missing decoder does."""

    init_signature = inspect.signature(LivePreviewCallback.__init__)
    instances: ClassVar[list["FakeLivePreview"]] = []
    fail_on_init: ClassVar[list[BaseException]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.init_signature.bind(self, **kwargs)
        if type(self).fail_on_init:
            raise type(self).fail_on_init[0]
        self.kwargs = kwargs
        self.models_at_init = len(FakeZImage.instances)
        self.save_to = kwargs["save_to"]
        self.variant = kwargs["variant"]
        self.events: list[tuple[str, object]] = []
        type(self).instances.append(self)

    def call_before_loop(self, **kwargs: Any) -> None:
        self.events.append(("before", self.save_to))

    def call_in_loop(self, *, denoised: mx.array | None = None, **kwargs: Any) -> None:
        self.events.append(("in", kwargs["t"]))
