"""
The mma.sync pieces of csrc/flash.cu, against torch.

Inputs are small integers, so every product and sum is exact in FP32: the result
must match bit for bit, and a lane that reads the wrong element cannot hide.
"""

from __future__ import annotations

import torch

from unica.cuda import module

WARP_SIZE = 32
HEAD_DIM = 128
KEYS_PER_TILE = 64


def small_ints(*shape: int) -> torch.Tensor:
    return torch.randint(-4, 5, shape, device="cuda").bfloat16()


def test_one_mma_matches_torch_exactly():
    torch.manual_seed(0)
    a, b = small_ints(16, 16), small_ints(8, 16)  # row-major B = the "col" operand
    d = torch.empty(16, 8, device="cuda")

    module("flash")["mma_16x8x16"](1, WARP_SIZE, a, b, d)
    torch.cuda.synchronize()

    expected = a.float() @ b.float().T
    assert torch.equal(d, expected), (d, expected)


def test_qk_tile_matches_torch_exactly():
    torch.manual_seed(1)
    q, k = small_ints(16, HEAD_DIM), small_ints(KEYS_PER_TILE, HEAD_DIM)  # |q·k| <= 16 · 128, exact
    s = torch.empty(16, KEYS_PER_TILE, device="cuda")

    module("flash")["qk_tile"](1, WARP_SIZE, q, k, s)
    torch.cuda.synchronize()

    expected = q.float() @ k.float().T
    assert torch.equal(s, expected), (s - expected).abs().max()


if __name__ == "__main__":
    test_one_mma_matches_torch_exactly()
    test_qk_tile_matches_torch_exactly()
    print("mma ok")
