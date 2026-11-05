"""
M4.23: V with one scale per token, folded into P (attention_int8pv_tma with V_TOKEN=1), against bench/emulate.py.

1. quantize_v(per_token=True) gives every pair of keys {log2 s, log2 s, 1 / s, 1 / s} with s = max |v_token| / 127.
2. The kernel is the emulator's v_token="float" within 0.1 %, on random V and on V with outlier tokens (Z-Image's),
   in the caller's [batch, tokens, heads, 128] layout too; with outliers it is far below the per-channel kernel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import unica  # noqa: E402
from emulate import TODAY, emulate, errors, reference  # noqa: E402

CONFIG = TODAY | dict(v_token="float")


def outlier_v(shape, seed=0, share=0.01, factor=100.0):
    """random V with `share` of its tokens `factor` times larger, as Z-Image's V (M4.22)"""
    torch.manual_seed(seed)
    v = torch.randn(shape, device="cuda")
    picked = torch.rand(shape[-2], device="cuda") < share
    v[..., picked, :] *= factor
    return v.bfloat16()


def test_token_scales():
    v = outlier_v((2, 3, 1000, 128))
    _, ones, tokens = unica.quantize_v(v, 2, per_token=True)
    s = (v.float().abs().amax(-1) / 127).clamp_min(1e-12)  # [b, h, n]
    pairs = torch.nn.functional.pad(s, (0, tokens.shape[-2] * 2 - s.shape[-1]))
    assert torch.equal(ones, torch.ones_like(ones))
    assert torch.allclose(tokens[..., 0:2].flatten(-2)[..., :1000], torch.log2(s), atol=1e-6)
    assert torch.allclose(tokens[..., 2:4].flatten(-2)[..., :1000], 1 / s, rtol=1e-6)
    assert not tokens.flatten(-2, -1)[..., 4 * 500 :].any()  # past the keys: zeros
    assert pairs.shape[-1] == 1024


def test_kernel_is_the_emulator():
    cases = [((1, 2, 1000, 128), None), ((2, 2, 777, 128), None), ((1, 4, 4096, 128), None),
             ((1, 2, 2048, 128), 0.01), ((1, 2, 4128, 128), 0.03)]
    for shape, share in cases:
        torch.manual_seed(1)
        q, k = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(2))
        v = outlier_v(shape, share=share) if share else torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        ref = reference(q, k, v)
        out = unica.attention(q, k, v, "attention_int8pv_tma", V_TOKEN=1).float()
        emulated = emulate(q, k, v, **CONFIG)
        plain = errors(unica.attention(q, k, v, "attention_int8pv_tma").float(), ref)["rel_l2"]
        assert errors(out, emulated)["rel_l2"] < 1e-3, (shape, share, errors(out, emulated))
        if share:
            assert errors(out, ref)["rel_l2"] < plain / 2, (shape, errors(out, ref), plain)
        print(f"{shape} outliers {share}: per token {100 * errors(out, ref)['rel_l2']:.3f} % "
              f"(emulator {100 * errors(emulated, ref)['rel_l2']:.3f} %), per channel {100 * plain:.3f} %")


def test_callers_layout():
    q, k = (torch.randn(2, 900, 3, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    v = outlier_v((2, 3, 900, 128)).transpose(1, 2).contiguous()  # [b, tokens, heads, 128]
    bhnd = [x.transpose(1, 2) for x in (q, k, v)]
    out = unica.attention(*bhnd, V_TOKEN=1)
    copied = unica.attention(*(x.contiguous() for x in bhnd), V_TOKEN=1)
    assert torch.equal(out, copied)


def sink_v(shape, seed=0):
    """random V whose tokens 1 and 2 are ~200× the others, as Z-Image's (M4.24)"""
    torch.manual_seed(seed)
    v = torch.randn(shape, device="cuda")
    v[..., 1:3, :] *= 200
    return v.bfloat16()


def test_outlier_scale_is_the_emulator():
    """M4.25: v_outliers' channel scale is the max of V with the picked keys zeroed, bit for bit"""
    for shape in ((1, 2, 1000, 128), (2, 3, 300, 128), (1, 1, 5, 128)):
        v = sink_v(shape)
        keys, scale = unica.v_outliers(v, 4)
        picked = v.float().abs().amax(-1).topk(min(4, shape[2]), -1).indices
        assert torch.equal(keys, picked)
        zeroed = v.float().scatter(-2, picked[..., None].expand(-1, -1, -1, 128), 0.0)
        assert torch.equal(scale, (zeroed.abs().amax(-2) / 127).clamp_min(1e-12)), shape


def test_outlier_kernel_is_the_emulator():
    """M4.25: attention_int8pv_tma with V_OUTLIERS=T is the emulator's v_outliers=T within 0.1 %"""
    for shape, sinks, t in (((1, 2, 1000, 128), False, 8), ((2, 2, 777, 128), True, 8), ((1, 4, 4128, 128), True, 2),
                            ((1, 2, 4096, 128), True, 5)):
        torch.manual_seed(2)
        q, k = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(2))
        v = sink_v(shape) if sinks else torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        ref = reference(q, k, v)
        out = unica.attention(q, k, v, "attention_int8pv_tma", V_OUTLIERS=t).float()
        emulated = emulate(q, k, v, **TODAY | dict(v_outliers=t))
        plain = errors(unica.attention(q, k, v, "attention_int8pv_tma").float(), ref)["rel_l2"]
        assert errors(out, emulated)["rel_l2"] < 1e-3, (shape, t, errors(out, emulated))
        if sinks:
            assert errors(out, ref)["rel_l2"] < plain / 2, (shape, errors(out, ref), plain)
        print(f"{shape} sinks {sinks} T {t}: outliers out {100 * errors(out, ref)['rel_l2']:.3f} % "
              f"(emulator {100 * errors(emulated, ref)['rel_l2']:.3f} %), per channel {100 * plain:.3f} %")


if __name__ == "__main__":
    test_token_scales()
    test_kernel_is_the_emulator()
    test_callers_layout()
    test_outlier_scale_is_the_emulator()
    test_outlier_kernel_is_the_emulator()
    print("v_token ok")
