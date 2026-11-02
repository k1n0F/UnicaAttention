"""The build and the loader: compile csrc/smoke.cu, launch axpy on torch tensors, check the result."""

import torch

from unica.cuda import module

n = 1_000_003  # not a multiple of the block: the bounds check matters
x = torch.randn(n, device="cuda")
y = torch.randn(n, device="cuda")
expected = 2.5 * x + y

module("smoke")["axpy"]((n + 255) // 256, 256, 2.5, x, y, n)
torch.cuda.synchronize()
torch.testing.assert_close(y, expected)  # not bit-exact: nvcc fuses a·x + y into one FMA, torch rounds twice
print("ok")
