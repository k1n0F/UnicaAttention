"""CUDA driver API through ctypes: nvcc builds a .cubin, the driver loads it, kernels run on torch's stream.

No C++ extension: nothing to rebuild for each torch version.
"""

import ctypes
import functools
import os
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ARCH = "sm_120a"

_cuda = ctypes.CDLL("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")


def _check(res):
    if res:
        msg = ctypes.c_char_p()
        _cuda.cuGetErrorString(res, ctypes.byref(msg))
        raise RuntimeError(f"CUDA driver error {res}: {msg.value.decode() if msg.value else '?'}")


def cubin(name):
    """csrc/<name>.cu -> build/<name>.cubin, rebuilt when the .cu is newer.

    Only the .cu file's own time is checked: a change in an included header does not trigger a rebuild.
    """
    src, out = ROOT / "csrc" / f"{name}.cu", ROOT / "build" / f"{name}.cubin"
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        out.parent.mkdir(exist_ok=True)
        cmd = ["nvcc", f"-arch={ARCH}", "-cubin", "-O3", "-std=c++17", "-lineinfo", "-o", str(out), str(src)]
        subprocess.run(cmd, check=True)
    return out.read_bytes()


@functools.cache
def _context():
    """The device's primary context, the one torch works in. Retained once per process and never released:
    torch keeps it alive anyway, and the driver frees it at exit."""
    _check(_cuda.cuInit(0))
    ctx = ctypes.c_void_p()
    _check(_cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), torch.cuda.current_device()))
    return ctx


@functools.cache
def module(name):
    """csrc/<name>.cu, loaded once per process and kept until exit, as Triton and torch keep theirs.
    Without unloading, repeated calls do not pile up. A changed .cu takes effect in a new process."""
    return Module(name)


class Module:
    """One .cubin loaded into torch's context. module["name"] gives a kernel. Use module(), not this directly."""

    def __init__(self, name):
        _check(_cuda.cuCtxSetCurrent(_context()))
        self._image = cubin(name)  # keeps the bytes alive while the module exists
        self._mod = ctypes.c_void_p()
        _check(_cuda.cuModuleLoadData(ctypes.byref(self._mod), self._image))

    def __getitem__(self, name):
        f = ctypes.c_void_p()
        _check(_cuda.cuModuleGetFunction(ctypes.byref(f), self._mod, name.encode()))
        return Kernel(f, self)


class Kernel:
    def __init__(self, f, module):
        self.f = f
        self.module = module  # a CUfunction is valid only while its CUmodule is loaded

    def __call__(self, grid, block, *args, smem=0):
        """Launch on torch's current stream. Tensors go as pointers, int as int32, float as float32;
        a ctypes value goes as is."""
        vals = [
            ctypes.c_void_p(a.data_ptr())
            if isinstance(a, torch.Tensor)
            else ctypes.c_int(a)
            if isinstance(a, int)
            else ctypes.c_float(a)
            if isinstance(a, float)
            else a
            for a in args
        ]
        params = (ctypes.c_void_p * len(vals))(*[ctypes.addressof(v) for v in vals])
        g = (*grid, 1, 1)[:3] if isinstance(grid, tuple) else (grid, 1, 1)
        b = (*block, 1, 1)[:3] if isinstance(block, tuple) else (block, 1, 1)
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        _check(_cuda.cuLaunchKernel(self.f, *g, *b, smem, stream, params, None))
