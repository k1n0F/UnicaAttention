"""The emulator's contract.

1. With every quantization step off, it matches the FP32 reference.
2. Its INT8 Q·Kᵀ is bit-exact with a real INT8 tensor-core MMA (torch._int_mm).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from emulate import emulate, errors, quant, reference

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
print("ok")
