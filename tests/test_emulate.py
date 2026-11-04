"""The emulator's contract.

1. With every quantization step off, it matches the FP32 reference.
2. Its INT8 Q·Kᵀ is bit-exact with a real INT8 tensor-core MMA (torch._int_mm).
3. Its e2m1 rounding keeps the grid, rounds ties to the even mantissa and saturates at 6.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from emulate import emulate, errors, block_scaled, quant, reference, to_e2m1

torch.manual_seed(0)
q, k, v = torch.randn(3, 1, 2, 1000, 128, device="cuda").unbind(0)  # 1000 tokens: the last KV tile is partial
ref = reference(q, k, v)
for smooth_k in (False, True):  # smoothing K is exact: softmax cancels q·mean
    e = errors(emulate(q, k, v, smooth_k=smooth_k, out_bf16=False), ref)
    assert e["rel_l2"] < 1e-5, (smooth_k, e)

(qh, _), (kh, _) = quant(q[0, 0], "int8", 128), quant(k[0, 0], "int8", 64)
emulated = (qh @ kh.T).to(torch.int32)
hardware = torch._int_mm(qh.to(torch.int8), kh.to(torch.int8).T)
assert torch.equal(emulated, hardware)

grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
assert torch.equal(to_e2m1(grid), grid) and torch.equal(to_e2m1(-grid), -grid)
ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
assert torch.equal(to_e2m1(ties), torch.tensor([0, 1, 1, 2, 2, 4, 4.0]))  # to the even mantissa
assert torch.equal(to_e2m1(torch.tensor([7.0, 1e6, -9.0])), torch.tensor([6.0, 6.0, -6.0]))
for fmt in ("nvfp4", "mxfp4"):  # a block that is already on the grid at a power-of-two scale comes back unchanged
    on_grid = grid.repeat(4).reshape(1, 1, 1, 32) * 0.25
    assert torch.equal(block_scaled(on_grid, fmt, -1), on_grid), fmt
print("ok")
