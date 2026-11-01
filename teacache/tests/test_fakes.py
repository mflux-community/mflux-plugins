"""The Qwen-Image fakes against the locked mflux source: they record and denoise the way QwenImage.generate_image
does, so the command tests built on them check what the real command writes."""

import ast
import inspect
import textwrap
from collections.abc import Callable
from typing import Any

import mlx.core as mx
from _fakes import FakeQwenImage, LoopingFakeQwenImage
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage


def _function_node(fn: Callable[..., Any]) -> ast.FunctionDef:
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))


def _keyword_names(fn: Callable[..., Any], callee: str) -> set[str]:
    calls = [
        node for node in ast.walk(_function_node(fn)) if isinstance(node, ast.Call) and ast.unparse(node.func) == callee
    ]
    assert len(calls) == 1, f"{fn.__qualname__} calls {callee} {len(calls)} times"
    return {keyword.arg for keyword in calls[0].keywords if keyword.arg is not None}


def _denoising_loop(fn: Callable[..., Any]) -> str:
    loops = [
        node
        for node in ast.walk(_function_node(fn))
        if isinstance(node, ast.For) and ast.unparse(node.iter) == "config.time_steps"
    ]
    assert len(loops) == 1, f"{fn.__qualname__} has {len(loops)} loops over config.time_steps"
    return ast.dump(loops[0])


def test_fake_qwen_records_what_the_real_method_records() -> None:
    """Bug: FakeQwenImage passes ImageUtil.to_image a keyword the real QwenImage.generate_image does not (for example
    generation_parameters copied from FakeZImage), so the sidecar tests pass against a record the real command never
    writes."""
    for callee in ("ImageUtil.to_image", "Config"):
        real = _keyword_names(QwenImage.generate_image, callee)
        fake = _keyword_names(FakeQwenImage.generate_image, callee)
        assert fake == real, f"{callee}: only the fake passes {fake - real}, only mflux passes {real - fake}"
    real_to_image = _keyword_names(QwenImage.generate_image, "ImageUtil.to_image")
    assert "negative_prompt" in real_to_image and "generation_parameters" not in real_to_image  # mflux 0.22.0


def test_the_looping_fake_runs_mfluxs_qwen_loop_body_verbatim() -> None:
    """Bug: the looping fake's denoising loop differs from QwenImage.generate_image's (one transformer call bound to a
    local, a missing ctx.in_loop or mx.eval, another KeyboardInterrupt path), so a real-library test passes on a loop
    mflux does not run."""
    assert _denoising_loop(LoopingFakeQwenImage.denoise) == _denoising_loop(QwenImage.generate_image)


class _Recorder:
    def __init__(self) -> None:
        self.events: list[str] = []

    def call_before_loop(self, **kwargs: Any) -> None:
        self.events.append("before_loop")

    def call_in_loop(self, *, t: int, **kwargs: Any) -> None:
        self.events.append(f"in_loop {t}")

    def call_after_loop(self, **kwargs: Any) -> None:
        self.events.append("after_loop")


def test_the_looping_fake_qwen_denoises_every_step_then_calls_the_decode_hook(monkeypatch) -> None:
    """Bug: the tiny QwenTransformer does not accept the production-shaped latents and encodings, or decode_hook runs
    before the after-loop callbacks, so the low-RAM tests check the transformer before mflux could have dropped it."""
    model = LoopingFakeQwenImage(model_config=ModelConfig.qwen_image())
    recorder = _Recorder()
    model.callbacks.register(recorder)
    seen: list[tuple[list[str], bool]] = []
    monkeypatch.setattr(
        LoopingFakeQwenImage,
        "decode_hook",
        staticmethod(lambda fake: seen.append((list(recorder.events), fake.transformer is not None))),
    )
    assert LoopingFakeQwenImage.transformer_ref is not None
    assert LoopingFakeQwenImage.transformer_ref() is model.transformer
    assert model.transformer.img_in.weight.dtype == mx.bfloat16

    image = model.generate_image(seed=3, prompt="p", num_inference_steps=4, width=64, height=64, negative_prompt="n")

    assert seen == [(["before_loop", "in_loop 0", "in_loop 1", "in_loop 2", "in_loop 3", "after_loop"], True)]
    assert (image.negative_prompt, image.steps, image.width, image.height) == ("n", 4, 64, 64)
