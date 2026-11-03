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
from emulate import quant  # noqa: E402

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


if __name__ == "__main__":
    test_matches_emulator_bit_for_bit()
    print("quantize ok")
