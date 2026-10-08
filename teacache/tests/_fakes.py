"""Fakes at the two boundaries this package owns nothing behind: the mflux model class (weights,
Metal) and mlx_teacache.apply_teacache (it rewires a real transformer). Every call is bound against
the real signature, so a keyword the real code doesn't take fails the test. Modelled on mflux's
tests/cli/helpers/command_fakes.py.

FakeTeaCache builds a real TeaCacheHandle and TeaCacheStats through mlx-teacache internals that are
not public API (mlx_teacache.handle.VariantPatch, the TeaCacheHandle constructor,
TeaCacheStats.record / finalize_last_generation). An mlx-teacache release can change them; then fix
this fake, not the plugin (AGENTS.md says the same)."""

import inspect
import warnings
from typing import Any, ClassVar

import mlx.core as mx
import mlx_teacache
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.compute_precision.compute_precision import ComputePrecision
from mflux.models.common.config.config import Config
from mflux.models.z_image.latent_creator import ZImageLatentCreator
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
from mflux.models.z_image.variants.z_image import ZImage
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil
from mlx_teacache import Provenance, StepDecision, TeaCacheHandle, TeaCacheStats
from mlx_teacache.handle import VariantPatch


class FakeZImage:
    """Stands in for ZImage. Construction is what ZImageCommand.load() does, so an empty
    `instances` means no weights were loaded. generate_image builds its GeneratedImage the way mflux
    0.22's ZImage.generate_image does (guidance forced to 0.0 on a model without guidance, the scheduler
    default resolved, a real Config, the real ImageUtil.to_image with the same keywords), so a save
    writes a real PNG and a real JSON sidecar with the fields mflux writes: steps, guidance, image_path,
    image_strength, pid_*, and float32 / compute_precision in generation_parameters.

    Two fields are simpler than the real ones, safe because no test here passes --lora or reads them
    back: lora_paths / lora_scales are the values passed to __init__ (the real LoRALoader resolves
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
        """No denoising loop and no callbacks: the image is made up, and FakeTeaCache commits its own record."""


class LoopingFakeZImage(FakeZImage):
    """A FakeZImage that runs mflux 0.22's denoising loop for real: the callback context (start, before_loop,
    in_loop, after_loop), self._predict(self.transformer), the scheduler steps. The transformer is a tiny real
    ZImageTransformer with random, never-loaded weights, so the real mlx_teacache.apply_teacache can patch this
    model, gate every step and commit its stats. The prompt encoding is a fixed random array (no text encoder)."""

    _predict = staticmethod(ZImage._predict)
    CAPTION_DIM = 32

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        mx.random.seed(0)
        self.transformer = ZImageTransformer(
            dim=64,
            n_layers=2,
            n_refiner_layers=1,
            n_heads=2,
            cap_feat_dim=self.CAPTION_DIM,
            axes_dims=[8, 12, 12],
            axes_lens=[64, 32, 32],
        )
        self.transformer.set_dtype(mx.bfloat16)  # loaded weights are bfloat16

    def denoise(self, *, config: Config, seed: int, prompt: str) -> None:
        latents = ZImageLatentCreator.create_noise(seed, config.height, config.width)
        text_encodings = mx.random.normal((7, self.CAPTION_DIM), key=mx.random.key(1)).astype(mx.bfloat16)
        # mflux encodes a negative prompt only above guidance 1.0.
        negative_encodings = text_encodings if config.guidance > 1.0 else None
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        predict = self._predict(self.transformer)
        for t in config.time_steps:
            sigma_t = config.scheduler.sigmas[t].reshape((1,))
            noise = predict(
                latents=latents,
                timestep=mx.ones_like(sigma_t) - sigma_t,
                sigmas=config.scheduler.sigmas,
                text_encodings=text_encodings,
                negative_encodings=negative_encodings,
                guidance=config.guidance,
            )
            latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
            ctx.in_loop(t, latents)
            mx.eval(latents)
        del predict
        ctx.after_loop(latents)


class LifecycleMarker:
    """Registered by FakeTeaCache where the real library registers its lifecycle callback."""

    def call_before_loop(self, **kwargs: Any) -> None:
        pass

    def call_after_loop(self, **kwargs: Any) -> None:
        pass


def new_handle(*, rel_l1_thresh: float, variant_id: str) -> TeaCacheHandle:
    """A real TeaCacheHandle around a real TeaCacheStats with nothing patched, as FakeTeaCache returns it."""
    handle = TeaCacheHandle(
        patch=VariantPatch(),
        stats=TeaCacheStats(),
        provenance=Provenance(source="builtin"),
        rel_l1_thresh=rel_l1_thresh,
    )
    handle.variant_id = variant_id  # type: ignore[attr-defined]
    return handle


def commit_generation(
    handle: TeaCacheHandle,
    *,
    active_steps: int,
    skipped_steps: frozenset[int],
    numerical_miss_step: int | None = None,
) -> None:
    """Records and commits one generation the way mlx-teacache 0.13.2's gate decides it: the first and the last
    active step are always "forced" (skip_first_n_steps = skip_last_n_steps = 1, _kernel/gate.py), so a
    skipped_steps entry there is ignored; then the one numerical miss, the skipped steps, "computed" for the rest."""
    for step in range(active_steps):
        if step == 0 or step == active_steps - 1:
            decision = "forced"
        elif step == numerical_miss_step:
            decision = "numerical-miss"
        elif step in skipped_steps:
            decision = "skipped"
        else:
            decision = "computed"
        handle.stats.record(
            StepDecision(step_idx=step, timestep=float(step), rel_l1=None, accumulated_distance=0.0, decision=decision)
        )
    handle.stats.finalize_last_generation(num_inference_steps=active_steps, cfg_was_active=False)


class FakeTeaCache:
    """Stands in for mlx_teacache.apply_teacache. Builds a real TeaCacheHandle around a real
    TeaCacheStats and, like the library's generate_image wrapper, commits one generation only when
    generate_image returns normally. Like the library's lifecycle it records ACTIVE steps (an
    image-to-image run starts at mflux's own init_time_step), with the decisions commit_generation
    describes, and it resolves the default threshold and the variant id from the real
    mlx_teacache.match_variant."""

    real_signature = inspect.signature(mlx_teacache.apply_teacache)

    def __init__(
        self,
        *,
        skipped_steps: frozenset[int] = frozenset({1, 2, 3}),
        numerical_miss_step: int | None = None,
        uncommitted_seeds: frozenset[int] = frozenset(),
        warning: Warning | None = None,
        error: Exception | None = None,
    ) -> None:
        self.skipped_steps = skipped_steps
        self.numerical_miss_step = numerical_miss_step
        self.uncommitted_seeds = uncommitted_seeds
        self.warning = warning
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def __call__(self, flux: Any, **kwargs: Any) -> TeaCacheHandle:
        self.real_signature.bind(flux, **kwargs)
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if self.warning is not None:
            warnings.warn(self.warning, stacklevel=2)
        variant = mlx_teacache.match_variant(flux.model_config, ZImage)
        assert variant is not None and variant.default_thresh is not None, "the plugin applied an unsupported model"
        requested = kwargs.get("rel_l1_thresh")
        handle = new_handle(
            rel_l1_thresh=variant.default_thresh if requested is None else requested, variant_id=variant.variant_id
        )
        flux.callbacks.register(LifecycleMarker())
        original = flux.generate_image

        def generate_image(**generate_kwargs: Any) -> GeneratedImage:
            image = original(**generate_kwargs)
            if generate_kwargs["seed"] not in self.uncommitted_seeds:
                steps = generate_kwargs["num_inference_steps"]
                active = (
                    steps
                    - Config(
                        model_config=flux.model_config,
                        num_inference_steps=steps,
                        image_path=generate_kwargs.get("image_path"),
                        image_strength=generate_kwargs.get("image_strength"),
                    ).init_time_step
                )
                commit_generation(
                    handle,
                    active_steps=active,
                    skipped_steps=self.skipped_steps,
                    numerical_miss_step=self.numerical_miss_step,
                )
            return image

        flux.generate_image = generate_image
        return handle
