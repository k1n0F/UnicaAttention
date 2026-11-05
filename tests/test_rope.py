"""
M4.18: RoPE inside the quantization, against bench/emulate.py's rope().

1. rope_stats gives each channel's min and max of the rotated x exactly, and its mean to float precision.
2. quantize_int8 and quantize_fp4_rows with rope= are, bit for bit, the emulator's quantization of rope(x) − mean.
3. attention(q, k, v, rope=table) is the emulator on the rotated q and k, within 0.1 %, for the three fast kernels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import unica  # noqa: E402
from emulate import TODAY, block_scaled, emulate, errors, quant, reference, rope, wan_rope_table  # noqa: E402
from test_fp4 import E2M1_VALUES, KERNEL_CONFIG  # noqa: E402

# a small latent of WAN's shape: 3 frames of 7 × 9 tokens = 189, past a tile edge and a stats block
T, H, W = 3, 7, 9
EMULATED = {
    "attention_int8pv_tma": dict(qk="int8", p_fmt="u8", v_fmt="int8", p_local=True, sum_rounded=True),
    "attention_fp4pv": KERNEL_CONFIG,
    "attention_fp4": KERNEL_CONFIG | dict(qk="nvfp4"),
}


def inputs(heads=3, batch=1, seed=0):
    torch.manual_seed(seed)
    shape = (batch, T * H * W, heads, 128)  # ComfyUI's layout, read through a transpose
    q, k, v = (torch.randn(shape, device="cuda").bfloat16().transpose(1, 2) for _ in range(3))
    return q, k, v, wan_rope_table(T, H, W)


def test_rope_stats():
    q, _, _, table = inputs()
    mean, low, high = unica.rope_stats(q, table)
    rotated = rope(q.float(), table)
    assert torch.equal(low, rotated.amin(-2)) and torch.equal(high, rotated.amax(-2))
    assert torch.allclose(mean, rotated.mean(-2), atol=1e-6), (mean - rotated.mean(-2)).abs().max()


def test_quantizers_rotate_like_the_emulator():
    q, k, _, table = inputs()
    mean = unica.rope_stats(k, table)[0]
    centred = rope(k.float(), table) - mean[..., None, :]
    k8, k_scale = unica.quantize_int8(k, mean, unica.K_SCALE_BLOCK, table)
    expected, expected_scale = quant(centred, "int8", unica.K_SCALE_BLOCK)
    assert torch.equal(k8.float(), expected), "int8 values"
    assert torch.equal(k_scale, expected_scale[..., :: unica.K_SCALE_BLOCK, 0]), "int8 scales"
    _, low, high = unica.rope_stats(q, table)
    q4, outer = unica.quantize_fp4_rows(q, None, 1, table, (low, high))
    nibbles = torch.stack([q4[..., :64] & 0xF, q4[..., :64] >> 4], -1).flatten(-2).long()
    scales = q4[..., 64:72].view(torch.float8_e4m3fn).float() * outer[..., None, None]
    got = E2M1_VALUES.to(q.device)[nibbles] * scales.repeat_interleave(16, -1)
    assert torch.equal(got, block_scaled(rope(q.float(), table), "nvfp4", -1)), "fp4"


def test_attention_with_rope_is_the_emulator():
    for batch in (1, 2):
        q, k, v, table = inputs(batch=batch, seed=batch)
        q_rot, k_rot = rope(q.float(), table), rope(k.float(), table)
        ref = reference(q_rot, k_rot, v)
        for kernel, config in EMULATED.items():
            out = unica.attention(q, k, v, kernel, rope=table).float()
            emulated = emulate(q_rot, k_rot, v, **config)
            assert errors(out, emulated)["rel_l2"] < 1e-3, (kernel, batch, errors(out, emulated))
            print(f"{kernel:22} batch {batch}: rel L2 {errors(out, ref)['rel_l2']:.3%}, "
                  f"emulator {errors(emulated, ref)['rel_l2']:.3%}, kernel vs emulator {errors(out, emulated)['rel_l2']:.3%}")


def test_smoothing_with_rope_is_the_emulator():
    """M4.21: Q smoothing on the rotated q (its block means and q̄ · k both from the values rotated in float)."""
    q, k, v, table = inputs(batch=2, seed=3)
    q = (q.float() + 2 * torch.randn(3, 1, 128, device="cuda")).bfloat16()
    q_rot, k_rot = rope(q.float(), table), rope(k.float(), table)
    out = unica.attention(q, k, v, "attention_fp4", rope=table, SMOOTH_Q=256).float()
    emulated = emulate(q_rot, k_rot, v, **EMULATED["attention_fp4"], smooth_q=256)
    assert errors(out, emulated)["rel_l2"] < 1e-3, errors(out, emulated)


def test_under_inference_mode():
    """ComfyUI samples under torch.inference_mode, where tensors have no version counter: the compact table's cache
    must work there too, and a second call with the same table must reuse it (M4.19, found in the first video run)."""
    q, k, v, _ = inputs()
    with torch.inference_mode():
        table = wan_rope_table(T, H, W)
        first = unica.attention(q, k, v, rope=table)
        kept = unica.compact_rope(table)
        assert unica.attention(q, k, v, rope=table).equal(first) and unica.compact_rope(table) is kept


if __name__ == "__main__":
    test_under_inference_mode()
    test_rope_stats()
    test_quantizers_rotate_like_the_emulator()
    test_attention_with_rope_is_the_emulator()
    test_smoothing_with_rope_is_the_emulator()
    print("rope ok")
