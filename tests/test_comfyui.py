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


if __name__ == "__main__":
    test_unica_mode_runs_unica_and_other_modes_pass_through()
    print("comfyui ok")
