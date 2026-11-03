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
from emulate import emulate, errors, reference  # noqa: E402

# The emulator settings each quantized kernel implements; the kernel must land on the emulator's
# output, not merely near FP32. The two differ only in P (fp16 in the kernel) and in summation order.
INT8_QK = dict(qk="int8", q_block=128, k_block=64, smooth_k=True)
EMULATED = {
    "attention_int8qk": INT8_QK,
    "attention_int8pv": dict(INT8_QK, p_fmt="u8", v_fmt="int8", p_local=True),
}
EMULATED["attention_int8pv_async"] = EMULATED["attention_int8pv"]
MAX_FROM_EMULATOR = 1e-3

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
# tile shapes bench/compare.py measures; each must be correct before its time means anything
TILE_VARIANTS = [
    {"WARPS_PER_BLOCK": 8},
    {"KEYS_PER_TILE": 32},
    {"WARPS_PER_BLOCK": 8, "KEYS_PER_TILE": 32},
    {"MIN_BLOCKS_PER_SM": 3},
]


def test_every_kernel_matches_fp32_reference_on_all_shapes():
    torch.manual_seed(0)
    for batch, heads, n_queries, n_keys in SHAPES:
        q = torch.randn(batch, heads, n_queries, 128, device="cuda").bfloat16()
        k, v = (torch.randn(batch, heads, n_keys, 128, device="cuda").bfloat16() for _ in range(2))
        ref = reference(q, k, v)
        floor = errors(ref.bfloat16().float(), ref)["rel_l2"]

        variants = [(kernel, {}) for kernel in unica.KERNELS] + [("attention_smem", t) for t in TILE_VARIANTS]
        for kernel, tile in variants:
            out = unica.attention(q, k, v, kernel=kernel, **tile)
            error = errors(out.float(), ref)["rel_l2"]
            name = kernel + "".join(f" {key}={value}" for key, value in tile.items())
            assert torch.isfinite(out).all(), name
            if kernel in unica.QUANTIZED:
                emulated = emulate(q, k, v, **EMULATED[kernel])
                from_emulator = errors(out.float(), emulated)["rel_l2"]
                predicted = errors(emulated, ref)["rel_l2"]
                print(f"{name:52} {(batch, heads, n_queries, n_keys)!s:22} rel L2 {error:.3%}, "
                      f"emulator predicts {predicted:.3%}, kernel vs emulator {from_emulator:.3%}")
                assert from_emulator < MAX_FROM_EMULATOR, (name, from_emulator)
            else:
                print(f"{name:52} {(batch, heads, n_queries, n_keys)!s:22} rel L2 {error:.3%}, floor {floor:.3%}")
                assert error < floor + MAX_ABOVE_FLOOR, (name, error, floor)


def test_async_loads_change_nothing_but_time():
    torch.manual_seed(1)
    for batch, heads, n_queries, n_keys in SHAPES:
        q = torch.randn(batch, heads, n_queries, 128, device="cuda").bfloat16()
        k, v = (torch.randn(batch, heads, n_keys, 128, device="cuda").bfloat16() for _ in range(2))
        inputs = unica.kernel_inputs(q, k, v, "attention_int8pv")
        sync = unica.attention(q, k, v, "attention_int8pv", inputs)
        assert torch.equal(unica.attention(q, k, v, "attention_int8pv_async", inputs), sync), (n_queries, n_keys)


def test_rejects_what_the_kernel_cannot_take():
    q = torch.randn(1, 1, 16, 64, device="cuda").bfloat16()  # head_dim 64
    for bad in (q, q.float()):
        try:
            unica.attention(bad, bad, bad)
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"accepted {bad.dtype} {tuple(bad.shape)}")


if __name__ == "__main__":
    test_every_kernel_matches_fp32_reference_on_all_shapes()
    test_async_loads_change_nothing_but_time()
    test_rejects_what_the_kernel_cannot_take()
    print("attention ok")
