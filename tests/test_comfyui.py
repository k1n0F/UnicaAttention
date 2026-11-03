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
        expected = unica.attention(*bhnd).transpose(1, 2).half()
        assert out.dtype == q.dtype and out.shape == q.shape and torch.equal(out, expected)
        assert not calls

        assert fake.attention(q, k, v, attention_mode="sageattn") is q and calls == ["sageattn"]
    finally:
        del sys.modules[fake.__name__]


def test_native_override_takes_self_attention_and_passes_the_rest_on():
    calls = []
    previous = lambda func, q, *args, **kwargs: calls.append(q.shape) or "previous"  # noqa: E731
    override = comfyui.native_override(previous)
    torch.manual_seed(1)
    b, s, h = 1, 300, 4
    q, k, v = (torch.randn(b, s, h * 128, device="cuda", dtype=torch.float16) for _ in range(3))
    bhnd = [x.view(b, s, h, 128).transpose(1, 2).bfloat16().contiguous() for x in (q, k, v)]
    expected = unica.attention(*bhnd)  # [b, h, s, 128]

    # ComfyUI's call for self-attention: [b, s, h · 128] in and out, plus its own keyword arguments
    out = override(None, q, k, v, heads=h, transformer_options={}, _inside_attn_wrapper=True)
    assert out.shape == q.shape and out.dtype == q.dtype
    assert torch.equal(out, expected.transpose(1, 2).reshape(b, s, h * 128).half())
    # skip_reshape / skip_output_reshape: [b, h, s, 128] both ways
    out = override(None, *(x.half() for x in bhnd), heads=h, skip_reshape=True, skip_output_reshape=True)
    assert torch.equal(out, expected.half())
    assert not calls

    text = torch.randn(b, 512, h * 128, device="cuda", dtype=torch.float16)
    assert override(None, q, text, text, heads=h) == "previous"  # cross-attention
    assert override(None, q, k, v, heads=h, mask=torch.ones(1, device="cuda")) == "previous"
    assert override(None, q, k, v, h) == "previous"  # heads passed by position: not ours to guess
    assert len(calls) == 3

    class Patcher:  # the slice of ComfyUI's ModelPatcher the node uses
        def __init__(self, options):
            self.model_options = {"transformer_options": options}

        def clone(self):
            return Patcher(dict(self.model_options["transformer_options"]))

    node = comfyui.UnicaAttentionNative()
    (once,) = node.patch(Patcher({"optimized_attention_override": previous}), True, 0, 2)
    (twice,) = node.patch(once, True, 0, 2)
    assert twice.model_options["transformer_options"]["optimized_attention_override"] is \
        once.model_options["transformer_options"]["optimized_attention_override"]  # not stacked on itself


if __name__ == "__main__":
    test_unica_mode_runs_unica_and_other_modes_pass_through()
    test_native_override_takes_self_attention_and_passes_the_rest_on()
    print("comfyui ok")
