"""Accuracy emulator.

Runs the algorithm the kernel will run, KV tile by KV tile, with every quantization step
switchable on its own, and measures the error against an FP32 reference. Accuracy decisions
are made here, in numbers, before any CUDA is written.

    uv run python bench/emulate.py [--heads 40] [--tokens 32760] [--bc 64]
"""

import argparse
import math

import torch
import torch.nn.functional as F

E4M3 = torch.float8_e4m3fn
E4M3_MAX = 448.0
LOG2E = 1.4426950408889634
MMA_K = 32  # tokens summed by one m16n8k32 P·V instruction
SKIP_ROWS = 64  # query rows of one block of the kernel: 4 warps × 16, which pass each tile's barrier together

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


E2M1_MAX = 6.0
# block-scaled formats: elements along the summed axis that share one scale
FP4_BLOCK = {"nvfp4": 16, "mxfp4": 32, "mxfp8": 32}  # mxfp8: e4m3, the only scaled form that takes e2m1 × e4m3


def to_e2m1(x):
    """Round to the nearest e2m1 value (±0, 0.5, 1, 1.5, 2, 3, 4, 6), ties to even, saturating at 6, as
    cvt.rn.satfinite.e2m1x2.f32 does. The grid step is 0.5 below 2, 1 below 4, 2 up to 6; on each piece an even
    multiple of the step is an even mantissa, so torch.round's ties to even are e2m1's."""
    a = x.abs()
    step = torch.where(a < 2, 0.5, torch.where(a < 4, 1.0, 2.0))
    return (torch.round(a / step) * step).clamp(max=E2M1_MAX).copysign(x)


def block_scaled(x, fmt, dim, outer=None, pow2=False):
    """x as a block-scaled FP4 MMA sees it: e2m1 values times one scale per FP4_BLOCK[fmt] elements along
    `dim`, the axis the MMA sums over. Returns the dequantized x̂ · s: the MMA applies the scales itself, with
    an f32 accumulator, so the emulator can multiply them in before an FP32 GEMM.

    nvfp4  scale = amax / 6 rounded to e4m3, times an FP32 outer scale. outer: None, one per head that puts the
           head's largest block at 448; "channel", the same per channel (every axis but `dim` kept); a number,
           fixed (the kernel's P, whose largest value is known). pow2: the scale is instead 2^ceil(log2(amax / 6))
           within ue4m3's 2^-9 .. 2^8, never saturating (M4.9)
    mxfp4  scale = 2^(floor(log2 amax) − 2), the OCP MX rule (6 = 1.5 · 2²); a block whose amax / scale lies in
           (6, 8) saturates at 6
    mxfp8  e4m3 values, scale = 2^(floor(log2 amax) − 8) (448 = 1.75 · 2⁸)
    """
    if fmt == "mxfp8":
        x = x.movedim(dim, -1)
        n = x.shape[-1]
        blocks = F.pad(x, (0, -n % 32)).unflatten(-1, (-1, 32))
        amax = blocks.abs().amax(-1, keepdim=True)
        s = torch.exp2(torch.floor(torch.log2(amax.clamp_min(2.0**-126))) - 8)
        x_hat = (blocks / s).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3).float() * s
        return x_hat.flatten(-2)[..., :n].movedim(-1, dim)
    block = FP4_BLOCK[fmt]
    x = x.movedim(dim, -1)
    n = x.shape[-1]
    blocks = F.pad(x, (0, -n % block)).unflatten(-1, (-1, block))
    amax = blocks.abs().amax(-1, keepdim=True)
    if fmt == "mxfp4":
        s = torch.exp2(torch.floor(torch.log2(amax.clamp_min(2.0**-126))) - 2)
    elif pow2:
        s = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-30) / E2M1_MAX / outer)).clamp(-9, 8)) * outer
    else:
        if outer is None or outer == "channel":
            over = (-3, -2, -1) if outer is None else (-2, -1)
            outer = (amax.amax(over, keepdim=True) / (E2M1_MAX * E4M3_MAX)).clamp_min(1e-30)
        s = (amax / E2M1_MAX / outer).clamp(max=E4M3_MAX).to(E4M3).float() * outer  # below e4m3's range: 0
    x_hat = to_e2m1(blocks / s.clamp_min(1e-30)) * s
    return x_hat.flatten(-2)[..., :n].movedim(-1, dim)


# m16n8k64 FP4 (LOG M4.8, measured): lane l holds k = 8 · (l % 4) + 32 · h + j of P's rows in register 2h (+1),
# j = 0..7, and gets the scores of keys 8 · b + 2 · (l % 4) + {0, 1}, b = 0..7, from Q·Kᵀ. Packing the scores of
# blocks 4h .. 4h + 3 in order into register 2h puts key 32h + 8 · (j / 2) + 2t + j % 2 at slot k = 32h + 8t + j.
FP4_SLOT_KEY = torch.tensor([32 * (k // 32) + 8 * (k % 8 // 2) + 2 * (k % 32 // 8) + k % 2 for k in range(64)])


def rope(x, freqs):
    """ComfyUI's apply_rope1 in float, as the kernels compute it (M4.19): with (c, s) of each pair rounded to fp16,
    out[2i] = c · x[2i] − s · x[2i + 1], out[2i + 1] = s · x[2i] + c · x[2i + 1], each product rounded, then the
    difference or sum (no fused multiply-add). x float [b, h, s, d]; freqs float32 [b or 1, s, 1, d / 2, 2, 2], a
    rotation per pair. Returns float, not rounded to bf16."""
    f = freqs[:, None, :, 0]  # [b or 1, 1, s, d / 2, 2, 2]
    c, s = f[..., 0, 0].half().float(), f[..., 1, 0].half().float()
    x0, x1 = x[..., 0::2], x[..., 1::2]
    return torch.stack([c * x0 - s * x1, s * x0 + c * x1], -1).flatten(-2)


def wan_rope_table(t, h, w, dim=128, theta=10000.0, device="cuda"):
    """WAN's RoPE table as ComfyUI builds it (EmbedND, axes [44, 42, 42] for d = 128) for a t × h × w latent: float32
    [1, t · h · w, 1, 64, 2, 2]."""
    axes = [dim - 4 * (dim // 6), 2 * (dim // 6), 2 * (dim // 6)]
    ids = torch.stack(torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij"), -1).reshape(-1, 3)
    tables = []
    for axis, size in enumerate(axes):
        omega = 1.0 / theta ** torch.linspace(0, (size - 2) / size, size // 2, dtype=torch.float64)
        angle = ids[:, axis, None].double() * omega  # [s, size / 2]
        tables.append(torch.stack([angle.cos(), -angle.sin(), angle.sin(), angle.cos()], -1).view(-1, size // 2, 2, 2))
    return torch.cat(tables, 1).float()[None, :, None].to(device)


def hadamard(n, device):
    """The n × n Hadamard matrix (Sylvester), scaled to be orthogonal: H · Hᵀ = I."""
    h = torch.ones(1, 1, device=device)
    while len(h) < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)])
    return h / len(h) ** 0.5


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
    sum_rounded=False,
    smooth_v=False,
    rotate_qk=None,
    p_terms=1,
    v_terms=1,
    p1_v1=True,
    skip_below=None,
    stats=None,
    fp4_slots=False,
    p_outer=None,
    v_outer=None,
    p_pow2=False,
    smooth_q=None,
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
    sum_rounded  the row sum over P as it goes into the MMA (rounded to p_fmt), not over P before
              rounding: numerator and denominator then see the same P
    rotate_qk None | 16 | 32 | 64 | 128: Q and K times the orthogonal Hadamard matrix, block-diagonal in blocks of
              this many channels of d, before quantization. Spreads a channel outlier over the block
    smooth_v  subtract the mean of V over tokens and add it to the output: a row of P sums to 1, so
              P·(V − μ) + μ = P·V
    p_terms, v_terms  1 or 2 block-scaled terms: x ≈ x₀ + x₁, x₁ the same format of the residual x − x₀.
              Each pair of terms is one more mma; p1_v1=False leaves out P₁·V₁
    skip_below  None | τ: a block of the kernel (SKIP_ROWS query rows) skips the softmax and P·V of a KV tile in
              which every row's P stays below τ · its running max, i.e. tile max − m < log2 τ in every row.
              Its scores are still computed: the max has to be known to decide
    stats     a dict that gets "skipped" and "pairs": skipped (block, tile) pairs and all of them
    fp4_slots the keys of every tile of 64 in the order of the k slots of the FP4 mma, as the kernel lays out V
              (FP4_SLOT_KEY), so that a block of 16 for a scale is the kernel's 16 keys. Needs bc = 64
    p_outer, v_outer  the outer scale of block_scaled for P and V
    p_pow2    P's block scales as powers of two (block_scaled's pow2)
    smooth_q  None | "head" | n: subtract the mean of Q over the head's queries ("head") or over each block of n
              queries before quantizing it, and add q̄ · k back to the scores, exactly (float, unquantized k): it
              varies along the keys, so softmax does not cancel it. With "head" it is one vector per key (M4.20)
    """
    q, k, v = q.float(), k.float(), v.float()
    if smooth_k:
        k = k - k.mean(-2, keepdim=True)
    if rotate_qk:  # (q·H)·(k·H) = q·H·Hᵀ·kᵀ = q·k: the scores do not change, the values do
        h = hadamard(rotate_qk, q.device)
        q, k = ((x.unflatten(-1, (-1, rotate_qk)) @ h).flatten(-2) for x in (q, k))
    v_mean = v.mean(-2, keepdim=True) if smooth_v else 0.0
    v = v - v_mean
    q_mean = None
    if smooth_q:
        if smooth_q == "head":
            q_mean = q.mean(-2, keepdim=True)
        else:
            n_q = q.shape[-2]
            blocks = F.pad(q, (0, 0, 0, -n_q % smooth_q)).unflatten(-2, (-1, smooth_q))
            sizes = torch.full((blocks.shape[-3],), float(smooth_q), device=q.device)
            sizes[-1] = n_q - smooth_q * (blocks.shape[-3] - 1)
            q_mean = (blocks.sum(-2) / sizes[:, None]).repeat_interleave(smooth_q, -2)[..., :n_q, :]
        q = q - q_mean
    k_exact = k  # for q̄ · k
    if qk in FP4_BLOCK:  # scales along d, inside the MMA
        q, k = block_scaled(q, qk, -1), block_scaled(k, qk, -1)
        sq = sk = torch.ones_like(q[..., :1])
    else:
        (q, sq), (k, sk) = quant(q, qk, q_block), quant(k, qk, k_block)
    key_valid = None
    if fp4_slots:  # a permutation inside each tile: the scores and P follow the keys, the sums do not change
        assert bc == 64, "the slot order is that of a 64-key tile"
        n = k.shape[-2]
        order = torch.arange(-(-n // 64) * 64, device=q.device).view(-1, 64)[:, FP4_SLOT_KEY.to(q.device)].flatten()
        k, sk, v, k_exact = (F.pad(x, (0, 0, 0, len(order) - n))[..., order, :] for x in (k, sk, v, k_exact))
        key_valid = order < n
    if v_fmt in FP4_BLOCK:  # scales per channel and block of keys, the axis P·V sums over
        v_parts, sv = [block_scaled(v, v_fmt, -2, v_outer)], 1.0
        if v_terms == 2:
            v_parts.append(block_scaled(v - v_parts[0], v_fmt, -2, v_outer))
        v = torch.stack(v_parts)  # [terms, ...]
    elif v_fmt:
        sv = (v.abs().amax(-2, keepdim=True) / QMAX[v_fmt]).clamp_min(1e-12)
        v = to_fmt(v / sv, v_fmt)
    else:
        sv = 1.0
    if v.dim() == q.dim():
        v = v.unsqueeze(0)  # one term
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
        if q_mean is not None:
            s = s + (q_mean @ k_exact[..., j : j + bc, :].transpose(-1, -2)) * scale
        if key_valid is not None:
            s = s.masked_fill(~key_valid[j : j + bc], -torch.inf)
        m_new = torch.maximum(m, s.amax(-1, keepdim=True))
        m_p = s.amax(-1, keepdim=True) if p_local else m_new
        alpha, beta = torch.exp2(m - m_new), torch.exp2(m_p - m_new)
        keep = 1.0
        if skip_below:
            negligible = (s.amax(-1) - m_new[..., 0]) < math.log2(skip_below)  # [..., rows]; never in the first tile
            rows = negligible.shape[-1]
            blocks = F.pad(negligible, (0, -rows % SKIP_ROWS), value=True).unflatten(-1, (-1, SKIP_ROWS)).all(-1)
            skipped = blocks.repeat_interleave(SKIP_ROWS, -1)[..., :rows, None]
            keep = (~skipped).float()  # a skipped tile leaves m, l and o as they were: its max is below m already
            if stats is not None:
                stats["skipped"] = stats.get("skipped", 0) + blocks.sum().item()
                stats["pairs"] = stats.get("pairs", 0) + blocks.numel()
        p = torch.exp2(s - m_p)
        row_sum = p.sum(-1, keepdim=True)  # the kernel sums P in FP32, before quantizing it
        p = p * p_mul
        p_parts = [p]
        if p_fmt in FP4_BLOCK:  # blocks of keys inside the tile
            p_parts = [block_scaled(p, p_fmt, -1, p_outer, p_pow2)]
            if p_terms == 2:
                p_parts.append(block_scaled(p - p_parts[0], p_fmt, -1, p_outer))
            p = sum(p_parts)
        elif p_fmt:
            p = p_parts[0] = to_fmt(p, p_fmt)
        if sum_rounded:
            row_sum = p.sum(-1, keepdim=True) / p_mul
        l = l * alpha + row_sum * beta * keep
        vt = v[..., j : j + bc, :]
        if len(p_parts) > 1 or len(vt) > 1:
            inner = sum(pi @ vj for i, pi in enumerate(p_parts) for jv, vj in enumerate(vt) if p1_v1 or i + jv < 2)
        elif acc16:
            vt = vt[0]
            inner = torch.zeros(o.shape, dtype=torch.half, device=o.device)
            for c in range(0, p.shape[-1], MMA_K):
                inner = (inner.float() + p[..., c : c + MMA_K] @ vt[..., c : c + MMA_K, :]).half()
            overflow |= torch.isinf(inner).any()
            inner = inner.float()
        else:
            inner = p @ vt[0]
        m, o = m_new, o * alpha + inner * beta * keep
    o = o * (sv / p_mul) / l + v_mean
    if overflow:
        print("  FP16 accumulator overflowed")
    return o.bfloat16().float() if out_bf16 else o


def negligible_tiles(q, k, taus, rows_per_block=(SKIP_ROWS, 16), bc=64, chunk=1024):
    """Share of (row block, KV tile) pairs whose tile could be skipped: every row's tile max lies below its
    running max (this tile included) by more than log2 τ. From exact FP32 scores, rows in chunks, so it is cheap
    next to emulate(). Returns {(rows, τ): share}."""
    q, k = q.float(), k.float()
    scale = q.shape[-1] ** -0.5 * LOG2E
    n_keys = k.shape[-2]
    counts = {(rows, tau): [0, 0] for rows in rows_per_block for tau in taus}
    for i in range(0, q.shape[-2], chunk):
        s = q[..., i : i + chunk, :] @ k.transpose(-1, -2) * scale
        tile_max = F.pad(s, (0, -n_keys % bc), value=-torch.inf).unflatten(-1, (-1, bc)).amax(-1)  # [..., rows, tiles]
        gap = tile_max - tile_max.cummax(-1).values
        for rows in rows_per_block:
            for tau in taus:
                negligible = (gap < math.log2(tau)).transpose(-1, -2)  # [..., tiles, rows]
                n = negligible.shape[-1]
                blocks = F.pad(negligible, (0, -n % rows), value=True).unflatten(-1, (-1, rows)).all(-1)
                counts[rows, tau][0] += blocks.sum().item()
                counts[rows, tau][1] += blocks.numel()
    return {key: skipped / total for key, (skipped, total) in counts.items()}


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
# M4: FP4 in one half of the tile or both, against today's kernel (the first line)
TODAY = dict(qk="int8", p_fmt="u8", v_fmt="int8", p_local=True, sum_rounded=True)
CONFIGS |= {
    "M4 today: INT8 QK + u8/s8 PV": TODAY,
    "M4 QK nvfp4 alone": dict(qk="nvfp4"),
    "M4 QK mxfp4 alone": dict(qk="mxfp4"),
    "M4 P nvfp4 alone, local": dict(p_fmt="nvfp4", p_local=True),
    "M4 V nvfp4 alone": dict(v_fmt="nvfp4"),
    "M4 a: QK nvfp4 + u8/s8 PV": TODAY | dict(qk="nvfp4"),
    "M4 a: QK mxfp4 + u8/s8 PV": TODAY | dict(qk="mxfp4"),
    "M4 b: INT8 QK + nvfp4 PV": TODAY | dict(p_fmt="nvfp4", v_fmt="nvfp4"),
    "M4 b: INT8 QK + mxfp4 PV": TODAY | dict(p_fmt="mxfp4", v_fmt="mxfp4"),
    "M4 c: nvfp4 QK + nvfp4 PV": TODAY | dict(qk="nvfp4", p_fmt="nvfp4", v_fmt="nvfp4"),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=40)
    ap.add_argument("--tokens", type=int, default=32760)
    ap.add_argument("--bc", type=int, default=64)
    ap.add_argument("--only", default="", help="run only the configs whose name contains this")
    ap.add_argument("--q-mul", type=float, default=1.0, help="sharper attention: q times this")
    args = ap.parse_args()

    torch.manual_seed(0)
    shape = (1, args.heads, args.tokens, 128)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    q = (q.float() * args.q_mul).bfloat16()
    ref = reference(q, k, v)
    print(f"shape {shape}, bf16 random inputs, q x {args.q_mul}, KV tile {args.bc}\n")
    print(f"{'config':38} {'rel L2':>8} {'cos':>9} {'max abs':>9}")
    for name, cfg in ((n, c) for n, c in CONFIGS.items() if args.only in n):
        e = errors(emulate(q, k, v, bc=args.bc, **cfg), ref)
        print(f"{name:38} {e['rel_l2']:8.2%} {e['cos']:9.6f} {e['max_abs']:9.2e}")
