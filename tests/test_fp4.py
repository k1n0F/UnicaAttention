"""
M4.8: the NVFP4 P·V path against bench/emulate.py.

1. quantize_v4 holds, after decoding, exactly the V of block_scaled(..., "nvfp4", outer="channel") with the keys of
   each tile in the FP4 mma's slot order.
2. quantize_fp4_rows (Q and K of attention_fp4) is block_scaled(x − mean, "nvfp4", -1) bit for bit.
3. attention_fp4pv and attention_fp4 are the emulator's configurations within 0.1 %: not bit for bit, since P̂ comes
   out of ex2.approx with an lg2.approx offset where the emulator divides.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import unica  # noqa: E402
from emulate import FP4_SLOT_KEY, TODAY, block_scaled, emulate, errors, reference  # noqa: E402

# attention_fp4pv in the emulator: INT8 Q·Kᵀ as today, P against the running max times 2^10 in NVFP4 (fixed outer
# scale 1), V in NVFP4 under a scale per channel, the keys of a tile in slot order, Σ P̂ as the denominator
KERNEL_CONFIG = TODAY | dict(p_fmt="nvfp4", v_fmt="nvfp4", p_local=False, p_mul=1024.0, p_outer=1.0,
                             v_outer="channel", fp4_slots=True)

E2M1_VALUES = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
SCALE_BLOCKS = 4  # e4m3 scales per channel and tile


def decode_v4(v4: torch.Tensor, outer: torch.Tensor) -> torch.Tensor:
    """uint8 [batch, heads, tiles, 128, 48] -> float [batch, heads, tiles · 64, 128], keys in slot order."""
    nibbles = torch.stack([v4[..., :32] & 0xF, v4[..., :32] >> 4], -1).flatten(-2).long()  # [..., tiles, dim, 64]
    values = E2M1_VALUES.to(v4.device)[nibbles]
    dims = torch.arange(128, device=v4.device)
    rows = dims % 8 + 8 * (dims // 32)
    scale_bytes = torch.stack([v4[..., rows, 32 + 4 * (dims // 8 % 4) + i] for i in range(SCALE_BLOCKS)], -1)
    scales = scale_bytes.view(torch.float8_e4m3fn).float() * outer[..., None, :, None]  # [..., tiles, dim, 4]
    v_hat = values * scales.repeat_interleave(16, -1)
    return v_hat.transpose(-1, -2).flatten(-3, -2)


def slot_order(v: torch.Tensor) -> torch.Tensor:
    """v [..., n, 128] padded with zeros to whole tiles and reordered inside each tile as the kernel lays it out."""
    n = v.shape[-2]
    tiles = -(-n // unica.FP4_TILE)
    order = (torch.arange(tiles * unica.FP4_TILE).view(tiles, -1)[:, FP4_SLOT_KEY]).flatten().to(v.device)
    return torch.nn.functional.pad(v.float(), (0, 0, 0, tiles * unica.FP4_TILE - n))[..., order, :]


def test_quantize_v4_is_the_emulator():
    torch.manual_seed(0)
    for n in (64, 1000, 4096):  # 1000: a partial last tile
        v = torch.randn(1, 3, n, 128, device="cuda", dtype=torch.bfloat16)
        v[0, 1] *= 1e-3  # a head far from 1, and channels of different sizes
        v[..., :8] *= 40
        v4, outer = unica.quantize_v4(v)
        expected = block_scaled(slot_order(v), "nvfp4", -2, "channel")
        got = decode_v4(v4, outer)
        assert torch.equal(got, expected), (n, (got - expected).abs().max().item())


def test_quantize_fp4_rows_is_the_emulator():
    torch.manual_seed(0)
    for n in (1, 64, 1000):
        x = torch.randn(1, 3, n, 128, device="cuda", dtype=torch.bfloat16)
        x[..., :4] *= 30  # outlier channels
        x[0, 2] *= 1e-3
        for mean in (None, x.mean(-2, dtype=torch.float32)):
            x4, outer = unica.quantize_fp4_rows(x, mean)
            nibbles = torch.stack([x4[..., :64] & 0xF, x4[..., :64] >> 4], -1).flatten(-2).long()
            scales = x4[..., 64:72].view(torch.float8_e4m3fn).float() * outer[..., None, None]
            got = E2M1_VALUES.to(x.device)[nibbles] * scales.repeat_interleave(16, -1)
            centred = x.float() - mean[..., None, :] if mean is not None else x.float()
            assert torch.equal(got, block_scaled(centred, "nvfp4", -1)), (n, mean is None)


def test_kernel_is_the_emulator():
    for shape, sharpness in (((1, 1, 64, 128), 1), ((1, 2, 1000, 128), 1), ((2, 2, 777, 128), 1),
                             ((1, 4, 4096, 128), 1), ((1, 2, 2048, 128), 4)):
        torch.manual_seed(0)
        q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        q = (q.float() * sharpness).bfloat16()
        ref = reference(q, k, v)
        for kernel, config in (("attention_fp4pv", KERNEL_CONFIG), ("attention_fp4", KERNEL_CONFIG | dict(qk="nvfp4"))):
            out = unica.attention(q, k, v, kernel).float()
            emulated = emulate(q, k, v, **config)
            assert not torch.isnan(out).any(), (kernel, shape)
            assert errors(out, emulated)["rel_l2"] < 1e-3, (kernel, shape, sharpness, errors(out, emulated))
            assert abs(errors(out, ref)["rel_l2"] - errors(emulated, ref)["rel_l2"]) < 1e-4, (kernel, shape, sharpness)


if __name__ == "__main__":
    test_quantize_v4_is_the_emulator()
    test_quantize_fp4_rows_is_the_emulator()
    test_kernel_is_the_emulator()
    print("fp4 ok")
