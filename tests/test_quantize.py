"""
The CUDA quantization against bench/emulate.py's quant(): bit for bit, values and scales.

The emulator's accuracy numbers are only a prediction of the kernel if both quantize identically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import unica

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from emulate import QMAX, quant, to_fmt  # noqa: E402

SHAPES = [(1, 2, 1000, 128), (2, 3, 77, 128), (1, 1, 128, 128)]  # partial last blocks, fewer tokens than a block


def test_matches_emulator_bit_for_bit():
    torch.manual_seed(0)
    for shape in SHAPES:
        x = (torch.randn(shape, device="cuda") * torch.rand(shape[-1], device="cuda") * 4).bfloat16()
        for block, mean in ((128, None), (64, x.mean(-2, dtype=torch.float32))):
            x8, scale = unica.quantize_int8(x, mean, block)
            centered = x.float() if mean is None else x.float() - mean[..., None, :]
            expected, expected_scale = quant(centered, "int8", block)

            assert torch.equal(x8.float(), expected), (shape, block, (x8.float() - expected).abs().max())
            per_token = scale.repeat_interleave(block, -1)[..., : shape[2], None]
            assert torch.equal(per_token, expected_scale), (shape, block)


def test_v_matches_emulator_bit_for_bit():
    torch.manual_seed(1)
    for shape in SHAPES:
        v = (torch.randn(shape, device="cuda") * torch.rand(shape[-1], device="cuda") * 4).bfloat16()
        v8, scale = unica.quantize_v(v)
        # bench/emulate.py's emulate(), v_fmt="int8"
        expected_scale = (v.float().abs().amax(-2, keepdim=True) / QMAX["int8"]).clamp_min(1e-12)
        expected = to_fmt(v.float() / expected_scale, "int8")

        assert torch.equal(scale, expected_scale.squeeze(-2)), shape
        assert torch.equal(v8.float(), expected), (shape, (v8.float() - expected).abs().max())


def p_order_slot(key: torch.Tensor) -> torch.Tensor:
    """csrc/flash.cu's p_order_slot: where key k of a group of 16 sits in the V tile."""
    k = key % 16
    return key - k + 2 * (k % 8) - k % 2 + 2 * (k // 8)


def test_v_layouts_hold_the_same_bytes_in_p_order():
    # layout 1: row p_order_slot(key); layout 2: the pairs of csrc/flash.cu v8_offset. Both padded to 64 keys.
    torch.manual_seed(2)
    for shape in SHAPES:
        v = torch.randn(shape, device="cuda").bfloat16()
        plain, scale = unica.quantize_v(v)
        n = shape[2]
        slot = p_order_slot(torch.arange(n, device="cuda"))
        dims = torch.arange(128, device="cuda")
        for layout in (1, 2):
            v8, layout_scale = unica.quantize_v(v, layout)
            assert v8.shape[2] == -(-n // unica.V8_PAD_KEYS) * unica.V8_PAD_KEYS and torch.equal(layout_scale, scale)
            flat = v8.reshape(*shape[:2], -1)
            if layout == 1:
                offset = slot[:, None] * 128 + dims
            else:
                offset = ((slot // 32 * 16 + slot % 16) * 256)[:, None] + dims // 8 * 16 + (slot % 32 // 16 * 8)[:, None] + dims % 8
            assert torch.equal(flat[..., offset], plain), (shape, layout)


if __name__ == "__main__":
    test_matches_emulator_bit_for_bit()
    test_v_matches_emulator_bit_for_bit()
    test_v_layouts_hold_the_same_bytes_in_p_order()
    print("quantize ok")
