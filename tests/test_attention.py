"""
unica.attention against the FP32 reference, on shapes that hit every edge of the tiling.

The output is bf16, and rounding it alone already costs ~0.1 % (the floor). The kernel may add
only a little on top: its own FP16 P·V error is 0.02 %.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import unica

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from emulate import errors, reference  # noqa: E402

# (batch, heads, n_queries, n_keys)
SHAPES = [
    (1, 1, 64, 64),  # one query block, one key tile, nothing partial
    (1, 2, 1000, 1000),  # both the last query block and the last key tile are partial
    (2, 3, 129, 129),  # one row and one key past a tile edge
    (1, 2, 1, 1),  # a single token
    (1, 2, 300, 77),  # cross-attention: fewer keys than queries, one partial key tile only
    (1, 4, 4096, 4096),  # long enough for the running max to rescale many times
]
MAX_ABOVE_FLOOR = 5e-4


def test_matches_fp32_reference_on_all_shapes():
    torch.manual_seed(0)
    for batch, heads, n_queries, n_keys in SHAPES:
        q = torch.randn(batch, heads, n_queries, 128, device="cuda").bfloat16()
        k, v = (torch.randn(batch, heads, n_keys, 128, device="cuda").bfloat16() for _ in range(2))
        ref = reference(q, k, v)

        out = unica.attention(q, k, v)
        error = errors(out.float(), ref)["rel_l2"]
        floor = errors(ref.bfloat16().float(), ref)["rel_l2"]
        print(f"{(batch, heads, n_queries, n_keys)!s:24} rel L2 {error:.3%}, bf16 floor {floor:.3%}")
        assert torch.isfinite(out).all()
        assert error < floor + MAX_ABOVE_FLOOR, (error, floor)


def test_rejects_what_the_kernel_cannot_take():
    q = torch.randn(1, 1, 16, 64, device="cuda").bfloat16()  # head_dim 64
    for bad in (q, q.float()):
        try:
            unica.attention(bad, bad, bad)
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"accepted {bad.dtype} {tuple(bad.shape)}")


if __name__ == "__main__":
    test_matches_fp32_reference_on_all_shapes()
    test_rejects_what_the_kernel_cannot_take()
    print("attention ok")
