"""Accuracy emulator.

Runs the algorithm the kernel will run, KV tile by KV tile, with every quantization step
switchable on its own, and measures the error against an FP32 reference. Accuracy decisions
are made here, in numbers, before any CUDA is written.

    uv run python bench/emulate.py [--heads 40] [--tokens 32760] [--bc 64]
"""

import argparse

import torch
import torch.nn.functional as F

E4M3 = torch.float8_e4m3fn
E4M3_MAX = 448.0
LOG2E = 1.4426950408889634
MMA_K = 32  # tokens summed by one m16n8k32 P·V instruction

torch.set_float32_matmul_precision("highest")  # no TF32: FP32 GEMMs stay exact enough to be a reference


def reference(q, k, v, chunk=256):
    """FP32 attention. Q goes in chunks, so the whole S is never materialised."""
    q, k, v = q.float(), k.float(), v.float()
    scale = q.shape[-1] ** -0.5
    out = torch.empty_like(q)
    for i in range(0, q.shape[-2], chunk):
        s = q[..., i : i + chunk, :] @ k.transpose(-1, -2) * scale
        out[..., i : i + chunk, :] = s.softmax(-1) @ v
    return out


def token_scale(x, block, qmax):
    """One scale per block of `block` tokens, over all of d: [..., N, D] -> [..., N, 1].

    The scale never varies along d, the axis Q·Kᵀ sums over. That is what lets a single
    INT32 MMA be scaled once at the end.
    """
    n = x.shape[-2]
    a = x.abs().amax(-1)
    a = F.pad(a, (0, -n % block)).unflatten(-1, (-1, block)).amax(-1, keepdim=True)
    a = a.expand(*a.shape[:-1], block).flatten(-2)[..., :n]
    return (a / qmax).clamp_min(1e-12).unsqueeze(-1)


QMAX = {"int8": 127.0, "u8": 255.0, "fp8": E4M3_MAX}


def to_fmt(x, fmt):
    """Round values already scaled into the format's range."""
    if fmt == "int8":
        return x.round().clamp(-127, 127)
    if fmt == "u8":  # P >= 0, so u8 x s8 MMA (native, same rate) gives P one more bit
        return x.round().clamp(0, 255)
    if fmt == "fp8":
        return x.to(E4M3).float()
    raise ValueError(fmt)


def quant(x, fmt, block):
    """x ≈ x̂ · s: values x̂ as they enter the MMA, and a scale s per token, [..., N, 1]."""
    if fmt is None:
        return x, torch.ones_like(x[..., :1])
    s = token_scale(x, block, QMAX[fmt])
    return to_fmt(x / s, fmt), s


def emulate(
    q,
    k,
    v,
    *,
    qk=None,
    q_block=128,
    k_block=64,
    smooth_k=True,
    p_fmt=None,
    v_fmt=None,
    p_local=False,
    acc16=False,
    p_mul=None,
    bc=64,
    out_bf16=True,
):
    """Tiled online-softmax attention, the way the kernel computes it.

    qk        None | "int8" | "fp8": how Q and K are quantized, one scale per token block
    smooth_k  subtract the mean of K over tokens: q·mean is constant per row, so softmax cancels it
    p_fmt     None | "int8" | "u8" | "fp8": P as it goes into the P·V MMA, after multiplying by p_mul
    v_fmt     None | "int8" | "fp8": V, one scale per channel (d is not summed in P·V)
    p_local   P relative to the row max of this tile, not the running max, so it fills the
              format's range. Corrected by β = exp2(m_tile − m) at the flush, which is allowed:
              one scale per KV tile
    acc16     P·V accumulates in FP16 inside a KV tile and is flushed into FP32 between tiles
    p_mul     P is multiplied by this before the MMA (default: the format's max). With acc16,
              p_mul·448·bc must stay below 65504 to rule out FP16 overflow in the worst case
    """
    q, k, v = q.float(), k.float(), v.float()
    if smooth_k:
        k = k - k.mean(-2, keepdim=True)
    (q, sq), (k, sk) = quant(q, qk, q_block), quant(k, qk, k_block)
    if v_fmt:
        sv = (v.abs().amax(-2, keepdim=True) / QMAX[v_fmt]).clamp_min(1e-12)
        v = to_fmt(v / sv, v_fmt)
    else:
        sv = 1.0
    if p_mul is None:
        p_mul = QMAX.get(p_fmt, 1.0)

    scale = q.shape[-1] ** -0.5 * LOG2E
    m = torch.full((*q.shape[:-1], 1), -torch.inf, device=q.device)
    l = torch.zeros_like(m)
    o = torch.zeros_like(q)
    overflow = torch.zeros((), dtype=torch.bool, device=q.device)
    for j in range(0, k.shape[-2], bc):
        # the kernel's formula: an MMA over d, then the outer product of the per-token scales.
        # For int8, q̂·k̂ is an integer with |q̂·k̂| ≤ 127²·128 < 2²⁴, so the FP32 GEMM gives exactly the INT32 MMA result
        kt, skt = k[..., j : j + bc, :].transpose(-1, -2), sk[..., j : j + bc, :].transpose(-1, -2)
        s = (q @ kt) * (sq * skt) * scale
        m_new = torch.maximum(m, s.amax(-1, keepdim=True))
        m_p = s.amax(-1, keepdim=True) if p_local else m_new
        alpha, beta = torch.exp2(m - m_new), torch.exp2(m_p - m_new)
        p = torch.exp2(s - m_p)
        l = l * alpha + p.sum(-1, keepdim=True) * beta  # the kernel sums P in FP32, before quantizing it
        p = p * p_mul
        if p_fmt:
            p = to_fmt(p, p_fmt)
        vt = v[..., j : j + bc, :]
        if acc16:
            inner = torch.zeros(o.shape, dtype=torch.half, device=o.device)
            for c in range(0, p.shape[-1], MMA_K):
                inner = (inner.float() + p[..., c : c + MMA_K] @ vt[..., c : c + MMA_K, :]).half()
            overflow |= torch.isinf(inner).any()
            inner = inner.float()
        else:
            inner = p @ vt
        m, o = m_new, o * alpha + inner * beta
    o = o * (sv / p_mul) / l
    if overflow:
        print("  FP16 accumulator overflowed")
    return o.bfloat16().float() if out_bf16 else o


def errors(o, ref):
    d = o - ref
    return {
        "rel_l2": (d.norm() / ref.norm()).item(),
        "cos": F.cosine_similarity(o.flatten(), ref.flatten(), dim=0).item(),
        "max_abs": d.abs().max().item(),
    }


CONFIGS = {
    "exact, bf16 output (floor)": {},
    "QK int8, Q/128 K/64 tokens": dict(qk="int8"),
    "QK int8, per token": dict(qk="int8", q_block=1, k_block=1),
    "QK int8, no K smoothing": dict(qk="int8", smooth_k=False),
    "QK fp8, Q/128 K/64 tokens": dict(qk="fp8"),
    "P fp8": dict(p_fmt="fp8"),
    "P int8": dict(p_fmt="int8"),
    "P int8, tile-local max": dict(p_fmt="int8", p_local=True),
    "P u8, tile-local max": dict(p_fmt="u8", p_local=True),
    "V fp8 per channel": dict(v_fmt="fp8"),
    "V int8 per channel": dict(v_fmt="int8"),
    "PV acc fp16, P x2": dict(acc16=True, p_mul=2),
    "INT8 QK + FP8 PV, acc fp32": dict(qk="int8", p_fmt="fp8", v_fmt="fp8"),
    "INT8 QK + FP8 PV, acc fp16, P x2": dict(qk="int8", p_fmt="fp8", v_fmt="fp8", acc16=True, p_mul=2),
    "INT8 QK + INT8 PV": dict(qk="int8", p_fmt="int8", v_fmt="int8"),
    "INT8 QK + INT8 PV, tile-local P": dict(qk="int8", p_fmt="int8", v_fmt="int8", p_local=True),
    "INT8 QK + u8/s8 PV, tile-local P": dict(qk="int8", p_fmt="u8", v_fmt="int8", p_local=True),
    "INT8 per token + u8/s8 PV, local P": dict(qk="int8", q_block=1, k_block=1, p_fmt="u8", v_fmt="int8", p_local=True),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=40)
    ap.add_argument("--tokens", type=int, default=32760)
    ap.add_argument("--bc", type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(0)
    shape = (1, args.heads, args.tokens, 128)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    ref = reference(q, k, v)
    print(f"shape {shape}, bf16 random inputs, KV tile {args.bc}\n")
    print(f"{'config':38} {'rel L2':>8} {'cos':>9} {'max abs':>9}")
    for name, cfg in CONFIGS.items():
        e = errors(emulate(q, k, v, bc=args.bc, **cfg), ref)
        print(f"{name:38} {e['rel_l2']:8.2%} {e['cos']:9.6f} {e['max_abs']:9.2e}")
