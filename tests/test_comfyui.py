"""
The ComfyUI node's attention wrapper (comfyui/__init__.py), without ComfyUI: a stand-in for
ComfyUI-WanVideoWrapper's wanvideo.modules.model takes its place in sys.modules.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import comfyui  # noqa: E402

import unica  # noqa: E402


def test_unica_mode_runs_unica_and_other_modes_pass_through():
    calls = []
    fake = types.ModuleType("ComfyUI-WanVideoWrapper.wanvideo.modules.model")
    fake.attention = lambda q, k, v, *args, attention_mode="sdpa", **kwargs: calls.append(attention_mode) or q
    sys.modules[fake.__name__] = fake
    try:
        assert comfyui.wrap_attention() == 1
        assert comfyui.wrap_attention() == 0  # already wrapped: not twice

        torch.manual_seed(0)
        # WanVideoWrapper's layout [batch, tokens, heads, 128], in fp16 as the "fp16_fast" loader gives it
        q, k, v = (torch.randn(1, 300, 4, 128, device="cuda", dtype=torch.float16) for _ in range(3))
        out = fake.attention(q, k, v, k_lens=None, attention_mode=comfyui.MODE, heads=4)
        bhnd = [x.transpose(1, 2).bfloat16().contiguous() for x in (q, k, v)]
        expected = unica.attention(*bhnd, **unica.PRESETS["per_token"]).transpose(1, 2).half()
        assert out.dtype == q.dtype and out.shape == q.shape and torch.equal(out, expected)
        assert not calls
        # audit: the settings travel in the mode's name, one per patched model
        for mode, preset in (("unica_per_channel", "per_channel"), ("unica_fast", "fast")):
            out = fake.attention(q, k, v, attention_mode=mode)
            expected = unica.attention(*bhnd, **unica.PRESETS[preset]).transpose(1, 2).half()
            assert torch.equal(out, expected), mode
        assert comfyui.mode_name("outliers", "accurate") == "unica_outliers" and comfyui.mode_settings("unica_x") is None
        assert not calls

        assert fake.attention(q, k, v, attention_mode="sageattn") is q and calls == ["sageattn"]

        # audit: a call unica cannot take falls back to sdpa, and so does every later one, instead of ending the render
        original, comfyui.unica_nhd = comfyui.unica_nhd, lambda *x: (_ for _ in ()).throw(ValueError("refused"))
        try:
            assert fake.attention(q, k, v, attention_mode=comfyui.MODE) is q and calls[-1] == "sdpa"
            comfyui.unica_nhd = original
            assert fake.attention(q, k, v, attention_mode=comfyui.MODE) is q and calls[-1] == "sdpa"
        finally:
            comfyui.unica_nhd = original
            comfyui.FAILED.clear()
    finally:
        del sys.modules[fake.__name__]


def test_native_override_takes_self_attention_and_passes_the_rest_on():
    calls = []
    previous = lambda func, q, *args, **kwargs: calls.append(q.shape) or "previous"  # noqa: E731
    override = comfyui.native_override(previous, unica.PRESETS["per_token"])
    torch.manual_seed(1)
    b, s, h = 1, 300, 4
    q, k, v = (torch.randn(b, s, h * 128, device="cuda", dtype=torch.float16) for _ in range(3))
    bhnd = [x.view(b, s, h, 128).transpose(1, 2).bfloat16().contiguous() for x in (q, k, v)]
    expected = unica.attention(*bhnd, **unica.PRESETS["per_token"])  # [b, h, s, 128]

    # ComfyUI's call for self-attention: [b, s, h · 128] in and out, plus its own keyword arguments
    out = override(None, q, k, v, heads=h, transformer_options={}, _inside_attn_wrapper=True)
    assert out.shape == q.shape and out.dtype == q.dtype
    assert torch.equal(out, expected.transpose(1, 2).reshape(b, s, h * 128).half())
    # skip_reshape / skip_output_reshape: [b, h, s, 128] both ways
    out = override(None, *(x.half() for x in bhnd), heads=h, skip_reshape=True, skip_output_reshape=True)
    assert torch.equal(out, expected.half())
    # heads (and the rest) by position, as Flux, LTX, Qwen-Image and Lumina call it: the same call
    out = override(None, *(x.half() for x in bhnd), h, None, None, True, True)
    assert torch.equal(out, expected.half())
    assert not calls

    text = torch.randn(b, 512, h * 128, device="cuda", dtype=torch.float16)
    assert override(None, q, text, text, heads=h) == "previous"  # cross-attention
    assert override(None, q, k, v, heads=h, mask=torch.ones(1, device="cuda")) == "previous"
    assert override(None, q, k, v, h, torch.ones(1, device="cuda")) == "previous"  # a mask by position
    assert override(None, q, k, v, h, None, None, False, False, "more") == "previous"  # more than ComfyUI's signature
    fewer_heads = torch.randn(b, s, 2 * 128, device="cuda", dtype=torch.float16)  # audit: GQA, k and v with 2 heads
    assert override(None, q, fewer_heads, fewer_heads, heads=h) == "previous"
    assert len(calls) == 5

    class Patcher:  # the slice of ComfyUI's ModelPatcher the node uses
        def __init__(self, options):
            self.model_options = {"transformer_options": options}

        def clone(self):
            return Patcher(dict(self.model_options["transformer_options"]))

    node = comfyui.UnicaAttentionNative()
    (once,) = node.patch(Patcher({"optimized_attention_override": previous}), True, 0, 2)
    (twice,) = node.patch(once, True, 0, 2, mode="fast")
    hook = lambda patcher: patcher.model_options["transformer_options"]["optimized_attention_override"]  # noqa: E731
    assert hook(twice).previous is previous  # patched again: a new override over the same previous, not stacked

    # audit: each patched model keeps its own settings, whichever node ran last; the fast mode runs attention_fp4
    # with Q smoothed (M4.21), v_scale does not apply
    (fast_model,) = node.patch(Patcher({}), True, 0, 2, v_scale="outliers", mode="fast")
    (token_model,) = node.patch(Patcher({}), True, 0, 2)
    fast = unica.attention(*bhnd, kernel="attention_fp4", SMOOTH_Q=512)
    assert torch.equal(hook(fast_model)(None, q, k, v, heads=h), fast.transpose(1, 2).reshape(b, s, h * 128).half())
    assert torch.equal(hook(token_model)(None, q, k, v, heads=h),
                       expected.transpose(1, 2).reshape(b, s, h * 128).half())


def test_measurement_count_runs_per_step_and_block():
    """The count is step · blocks + block + 1 within a run. Calls outside the numbered blocks (Lumina's context
    refiners: sigmas, no block_index) are not counted and do not break it (Z-Image, Moss crashed on them)."""
    m = comfyui.Measurement("test")
    m.every = 1000  # counting runs only while measuring
    sigma = lambda s: {"sigmas": torch.tensor([s])}  # noqa: E731
    seen = []
    for s in (1.0, 0.5):  # two steps of a 3-block model, each preceded by a refiner call
        m.due(sigma(s))
        for block in range(3):
            m.due(sigma(s) | {"block_index": block, "total_blocks": 3})
            seen.append(m.calls)
    assert seen == [1, 2, 3, 4, 5, 6], seen
    m.due(sigma(1.0) | {"block_index": 0, "total_blocks": 3})  # sigma rises: a new run
    assert m.calls == 1

    # a model that numbers no blocks (LTX: sigmas, never a block index) is counted call by call (M4.23 lost it)
    ltx = comfyui.Measurement("test")
    ltx.every = 2
    assert [ltx.due(sigma(1.0)) for _ in range(4)] == [False, True, False, True]

    # off, it reads nothing: a GPU sigma would cost a device-to-host wait on every call (M4.28)
    class Untouchable:
        def max(self):
            raise AssertionError("sigma read while not measuring")
    off = comfyui.Measurement("test")
    assert not off.due({"sigmas": Untouchable(), "block_index": 0})


if __name__ == "__main__":
    test_unica_mode_runs_unica_and_other_modes_pass_through()
    test_native_override_takes_self_attention_and_passes_the_rest_on()
    test_measurement_count_runs_per_step_and_block()
    print("comfyui ok")
