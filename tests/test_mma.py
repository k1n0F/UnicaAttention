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
LOG2E = 1.4426950408889634


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


def test_one_int8_mma_matches_torch_exactly():
    torch.manual_seed(4)
    a = torch.randint(-127, 128, (16, 32), device="cuda", dtype=torch.int8)  # the full int8 range: int32 is exact
    b = torch.randint(-127, 128, (8, 32), device="cuda", dtype=torch.int8)
    d = torch.empty(16, 8, device="cuda", dtype=torch.int32)

    module("flash")["mma_s8_16x8x32"](1, WARP_SIZE, a, b, d)
    torch.cuda.synchronize()

    expected = (a.double() @ b.double().T).int()  # |sum| <= 127² · 32 < 2^53: float64 is exact
    assert torch.equal(d, expected), (d - expected).abs().max()


def test_qk_tile_matches_torch_exactly():
    torch.manual_seed(1)
    q, k = small_ints(16, HEAD_DIM), small_ints(KEYS_PER_TILE, HEAD_DIM)  # |q·k| <= 16 · 128, exact
    s = torch.empty(16, KEYS_PER_TILE, device="cuda")

    module("flash")["qk_tile"](1, WARP_SIZE, q, k, s)
    torch.cuda.synchronize()

    expected = q.float() @ k.float().T
    assert torch.equal(s, expected), (s - expected).abs().max()


def test_softmax_stats_match_logsumexp():
    n_keys = 1024  # 16 tiles: the running max is rescaled many times
    scale = HEAD_DIM**-0.5
    torch.manual_seed(2)
    for q_gain in (1.0, 30.0):  # x30: up to 2^174 > FP32 max 2^128, unless the max is subtracted first
        q = (torch.randn(16, HEAD_DIM, device="cuda") * q_gain).bfloat16()
        k = torch.randn(n_keys, HEAD_DIM, device="cuda").bfloat16()
        lse = torch.empty(16, device="cuda")

        module("flash")["softmax_stats"](1, WARP_SIZE, q, k, lse, n_keys, scale * LOG2E)
        torch.cuda.synchronize()

        expected = torch.logsumexp(q.float() @ k.float().T * scale, dim=-1) * LOG2E  # log2-sum-exp2
        torch.testing.assert_close(lse, expected, rtol=1e-5, atol=1e-4)


def test_attention_warp_matches_fp32():
    n_keys = 1024
    torch.manual_seed(3)
    q, k, v = (torch.randn(n, HEAD_DIM, device="cuda").bfloat16() for n in (16, n_keys, n_keys))
    out = torch.empty(16, HEAD_DIM, device="cuda")

    module("flash")["attention_warp"](1, WARP_SIZE, q, k, v, out, n_keys, HEAD_DIM**-0.5 * LOG2E)
    torch.cuda.synchronize()

    expected = torch.softmax(q.float() @ k.float().T * HEAD_DIM**-0.5, dim=-1) @ v.float()
    rel_l2 = ((out - expected).norm() / expected.norm()).item()
    print(f"attention_warp rel L2 vs FP32: {rel_l2:.3%}")
    assert rel_l2 < 5e-4, rel_l2  # fp16 P gives 0.02 %; bf16 P would give 0.16 % and fail here


if __name__ == "__main__":
    test_one_mma_matches_torch_exactly()
    test_one_int8_mma_matches_torch_exactly()
    test_qk_tile_matches_torch_exactly()
    test_softmax_stats_match_logsumexp()
    test_attention_warp_matches_fp32()
    print("mma ok")
