"""With every quantization step off, the emulator must match the FP32 reference."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from emulate import emulate, errors, reference

torch.manual_seed(0)
q, k, v = torch.randn(3, 1, 2, 1000, 128, device="cuda").unbind(0)  # 1000 tokens: the last KV tile is partial
ref = reference(q, k, v)
for smooth_k in (False, True):  # smoothing K is exact: softmax cancels q·mean
    e = errors(emulate(q, k, v, smooth_k=smooth_k, out_bf16=False), ref)
    assert e["rel_l2"] < 1e-5, (smooth_k, e)
print("ok")
