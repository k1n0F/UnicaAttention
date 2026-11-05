"""
UnicaAttention: attention kernel made for the RTX 5080 (sm_120).

    out = unica.attention(q, k, v)  # [batch, heads, tokens, 128] bf16, like SDPA's default layout
"""

from __future__ import annotations

import ctypes
import math
import weakref

import torch

from unica.cuda import UINT8, UINT16, module, tensor_map_2d

HEAD_DIM = 128
WARP_SIZE = 32
ROWS_PER_WARP = 16  # these three must match csrc/flash.cu
Q_SCALE_BLOCK = 128
K_SCALE_BLOCK = 64
LOG2E = 1.4426950408889634

# Every variant in csrc/flash.cu, oldest first. All take the same grid and block; bench/compare.py
# times them side by side. The quantized ones trade exactness for speed: their error is measured
# against bench/emulate.py, which predicts it.
KERNELS = ("attention", "attention_smem", "attention_int8qk", "attention_int8pv", "attention_int8pv_async",
           "attention_int8pv_pipe", "attention_int8pv_tma", "attention_fp4pv", "attention_fp4")
QUANTIZED = {"attention_int8qk", "attention_int8pv", "attention_int8pv_async", "attention_int8pv_pipe",
             "attention_int8pv_tma", "attention_fp4pv", "attention_fp4"}
# INT8 Q·Kᵀ, u8 × s8 P·V: rel L2 1.73 % against FP32 on random inputs (SageAttention 2: 3.9 %). The exact
# path is attention_smem, at 0.17 %, the bf16 output floor.
DEFAULT_KERNEL = "attention_int8pv_tma"

# The tile shape csrc/flash.cu is built with when nothing overrides it (its UNICA_* macros). FAST_EXP2=0
# brings back exp2f in the u8 kernels, to measure M3.5; LAZY_RESCALE=0 the multiply of out in every tile
# (M3.9); Q_SMEM=0 Q in registers in attention_int8pv_pipe (M3.13); SUM_ROUNDED=0 the denominator over P
# before rounding (M3.15); V_PAIRS=0 V8 in P's order but unpaired for attention_int8pv_tma (M3.17); TRACE=1
# records a timeline (bench/timeline.py). SMOOTH_Q=n (attention_fp4, a multiple of STATS_TOKENS) subtracts Q's mean
# over each block of n queries before quantizing it and adds q̄ · k back exactly (M4.21); BIAS_CHUNK tiles of that q̄ · k
# per copy.
TILE = {"WARPS_PER_BLOCK": 4, "KEYS_PER_TILE": 64, "MIN_BLOCKS_PER_SM": 1, "FAST_EXP2": 1, "LAZY_RESCALE": 1,
        "Q_SMEM": 1, "SUM_ROUNDED": 1, "V_PAIRS": 1, "TRACE": 0, "STAGES": 2, "BULK_COPY": 1, "SMOOTH_Q": 0, "BIAS_CHUNK": 32}
# M4.17: kernels that read q, k, v and write the output in the caller's layout (TokenLayout), e.g. ComfyUI's
# [batch, tokens, heads, 128] seen as [batch, heads, tokens, 128] through a transpose. The others get contiguous copies.
LAYOUT_KERNELS = {"attention_int8pv_tma", "attention_fp4pv", "attention_fp4"}


class TokenLayout(ctypes.Structure):
    """csrc/flash.cu TokenLayout: strides in elements of a [batch, heads, tokens, 128] tensor, and its heads."""

    _fields_ = [("batch", ctypes.c_longlong), ("head", ctypes.c_longlong), ("token", ctypes.c_longlong),
                ("heads", ctypes.c_int)]


def token_layout(x: torch.Tensor) -> TokenLayout:
    return TokenLayout(x.stride(0), x.stride(1), x.stride(2), x.shape[1])


def readable(x: torch.Tensor) -> torch.Tensor:
    """x as it is if the kernels can read it in place (last dimension contiguous, every token on a 16-byte boundary,
    as the 16-byte loads need), else a contiguous copy."""
    in_place = (x.stride(-1) == 1 and x.data_ptr() % 16 == 0
                and all(x.stride(i) % 8 == 0 or x.shape[i] == 1 for i in range(3)))
    return x if in_place else x.contiguous()



def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel: str = DEFAULT_KERNEL,
    inputs: tuple | None = None,
    rope: torch.Tensor | None = None,
    **tile: int,
) -> torch.Tensor:
    """softmax(q · kᵀ / sqrt(d)) · v, non-causal. q is [batch, heads, n_queries, 128]; k and v are
    [batch, heads, n_keys, 128], so cross-attention with a different n_keys works too.

    For tests and benchmarks: kernel picks a variant from KERNELS, and tile overrides entries of TILE,
    e.g. KEYS_PER_TILE=32; each combination is its own cubin. inputs is what kernel_inputs() returned for
    these q, k, v: the kernel then runs alone, without quantizing again.

    rope (M4.18, self-attention only): q and k come before RoPE, and rope is ComfyUI's table, float32
    [batch or 1, tokens, 1, 64, 2, 2] (apply_rope1's freqs_cis). The quantization rotates them in float on the way in,
    so the rotated q and k are never written, nor rounded to bf16."""
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}, expected one of {KERNELS}")
    if unknown := tile.keys() - TILE.keys():
        raise ValueError(f"unknown tile settings {sorted(unknown)}, expected some of {sorted(TILE)}")
    if not (q.dtype == k.dtype == v.dtype == torch.bfloat16):
        raise TypeError(f"q, k, v must be bfloat16, got {q.dtype}, {k.dtype}, {v.dtype}")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q, k, v must be on the GPU")
    if q.dim() != 4 or q.shape[-1] != HEAD_DIM:
        raise ValueError(f"q must be [batch, heads, tokens, {HEAD_DIM}], got {tuple(q.shape)}")
    if k.shape != v.shape or k.shape[:2] != q.shape[:2] or k.shape[-1] != HEAD_DIM:
        raise ValueError(f"k and v must be [batch, heads, n_keys, {HEAD_DIM}] like q, "
                         f"got q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)}")
    batch, heads, n_queries, _ = q.shape
    n_keys = k.shape[2]
    if n_keys == 0:
        raise ValueError("k and v must hold at least one token")
    if rope is not None:
        if kernel not in LAYOUT_KERNELS:
            raise ValueError(f"rope= needs one of {sorted(LAYOUT_KERNELS)}, got {kernel!r}")
        rope = rope_table(rope, batch, n_queries, n_keys)

    if kernel in LAYOUT_KERNELS:  # read in place, and write the output in q's layout (empty_like keeps its strides)
        q, k, v = readable(q), readable(k), readable(v)
        out = torch.empty_like(q)
        layout_arg = (token_layout(out),)
    else:  # these index plain [heads][tokens][128] arrays; for an already contiguous tensor this is free
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        out = torch.empty_like(q)
        layout_arg = ()
    defines = {f"UNICA_{key}": value for key, value in tile.items() if TILE[key] != value}
    warps = tile.get("WARPS_PER_BLOCK", TILE["WARPS_PER_BLOCK"])
    grid = (math.ceil(n_queries / (warps * ROWS_PER_WARP)), batch * heads)
    scale_log2 = HEAD_DIM**-0.5 * LOG2E
    if inputs is None:
        inputs = kernel_inputs(q, k, v, kernel, rope, **tile)
    module("flash", **defines)[kernel](grid, warps * WARP_SIZE, *inputs, out, n_queries, n_keys, scale_log2, *layout_arg)
    return out


def kernel_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kernel: str, rope: torch.Tensor | None = None,
                  **tile: int) -> tuple:
    """The arguments `kernel` takes before its output: q, k, v as they are, or quantized. The quantizers read q, k, v
    in any layout readable() accepts. Apart from attention() so that a benchmark can time quantization and kernel
    separately. tile:
    the settings the kernel is built with (TILE), which decide the boxes and the V8 layout of the TMA kernel. rope: see
    attention(); q and k are then taken before RoPE, and the K mean and the FP4 scales are those of the rotated values."""
    if kernel not in QUANTIZED:
        return q, k, v
    if kernel == "attention_fp4":  # M4.12: Q and K in NVFP4 too
        settings = TILE | tile
        keys, smooth = settings["KEYS_PER_TILE"], settings["SMOOTH_Q"]
        q_mean = None
        if smooth:  # M4.21: Q's means per block of `smooth` queries, and the stats of K in the same one pass
            q_mean, q_low, q_high = rope_stats(q, rope, smooth)
            k_mean, k_low, k_high = rope_stats(k, rope)
            q4, q_outer = quantize_fp4_rows(q, q_mean, 1, rope, (q_low, q_high), smooth)
            k4, k_outer = quantize_fp4_rows(k, k_mean, keys, rope, (k_low, k_high))
        elif rope is None:
            q4, q_outer = quantize_fp4_rows(q, None)
            k4, k_outer = quantize_fp4_rows(k, k.mean(-2, dtype=torch.float32), keys)  # softmax cancels the mean
        else:  # one pass of statistics over the rotated values instead of k.mean and aminmax
            _, q_low, q_high = rope_stats(q, rope)
            k_mean, k_low, k_high = rope_stats(k, rope)
            q4, q_outer = quantize_fp4_rows(q, None, 1, rope, (q_low, q_high))
            k4, k_outer = quantize_fp4_rows(k, k_mean, keys, rope, (k_low, k_high))
        qk_outer = (q_outer * k_outer).contiguous()
        bias = ctypes.c_void_p(0) if q_mean is None else smooth_bias(k, k_mean, q_mean, qk_outer, k4.shape[2], rope)
        v4, v_outer = quantize_v4(v)
        k_map = tensor_map_2d(k4, UINT8, FP4_ROW_BYTES, k4.numel() // FP4_ROW_BYTES, FP4_ROW_BYTES, FP4_ROW_BYTES, keys)
        v_map = tensor_map_2d(v4, UINT8, V4_ROW_BYTES, v4.numel() // V4_ROW_BYTES, V4_ROW_BYTES, V4_ROW_BYTES, HEAD_DIM)
        return k_map, v_map, k4, v4, q4, qk_outer, v_outer, bias
    q8, q_scale = quantize_int8(q, None, Q_SCALE_BLOCK, rope)
    k_mean = k.mean(-2, dtype=torch.float32) if rope is None else rope_stats(k, rope)[0]  # softmax cancels the mean
    k8, k_scale = quantize_int8(k, k_mean, K_SCALE_BLOCK, rope)
    if kernel == "attention_int8qk":
        return q8, k8, q_scale, k_scale, v
    if kernel == "attention_fp4pv":  # V4: one box of HEAD_DIM rows of V4_ROW_BYTES is one tile
        keys = (TILE | tile)["KEYS_PER_TILE"]
        v4, v_outer = quantize_v4(v)
        k_map = tensor_map_2d(k8, UINT8, HEAD_DIM, k8.numel() // HEAD_DIM, HEAD_DIM, HEAD_DIM + 16, keys)
        v_map = tensor_map_2d(v4, UINT8, V4_ROW_BYTES, v4.numel() // V4_ROW_BYTES, V4_ROW_BYTES, V4_ROW_BYTES, HEAD_DIM)
        return k_map, v_map, v4, q8, q_scale, k_scale, v_outer
    if kernel != "attention_int8pv_tma":
        return q8, k8, q_scale, k_scale, *quantize_v(v)
    # TMA boxes 16 bytes wider than the rows: the tiles land padded, as the kernel's ldmatrix expects
    settings = TILE | tile
    keys, pairs = settings["KEYS_PER_TILE"], settings["V_PAIRS"]
    v8, v_scale = quantize_v(v, 2 if pairs else 1)
    k_map = tensor_map_2d(k8, UINT8, HEAD_DIM, k8.numel() // HEAD_DIM, HEAD_DIM, HEAD_DIM + 16, keys)
    if pairs:  # rows of 2 · HEAD_DIM bytes, as uint16 so the box (272 bytes) stays within 256 elements
        v_map = tensor_map_2d(v8, UINT16, HEAD_DIM, v8.numel() // (2 * HEAD_DIM), 2 * HEAD_DIM, HEAD_DIM + 8, keys // 2)
    else:
        v_map = tensor_map_2d(v8, UINT8, HEAD_DIM, v8.numel() // HEAD_DIM, HEAD_DIM, HEAD_DIM + 16, keys)
    return k_map, v_map, q8, q_scale, k_scale, v_scale  # the maps first: they must be 64-byte aligned


QUANTIZE_THREADS = 256  # must match csrc/flash.cu
CHUNK = 8  # bf16 per thread in quantize_v


def quantize_int8(x: torch.Tensor, mean: torch.Tensor | None, block: int,
                  rope: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """x − mean ≈ x8 · scale, one scale per `block` tokens over all of d. x is contiguous bf16
    [batch, heads, N, 128], mean float32 [batch, heads, 128] or None. Returns int8 [batch, heads, N, 128]
    and float32 scales [batch, heads, ceil(N / block)]. The scale never varies along d, the axis Q·Kᵀ sums
    over, so the int32 mma result is scaled once at the end. Bit for bit bench/emulate.py's quant(). rope: x is
    rotated first (rope_table), then the mean (of the rotated x) subtracted."""
    batch, heads, n, _ = x.shape
    x = readable(x)
    x8 = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scale = torch.empty(batch, heads, math.ceil(n / block), device=x.device)
    grid = (scale.shape[-1], batch * heads)
    mean_arg = mean.contiguous() if mean is not None else ctypes.c_void_p(0)
    module("flash")["quantize_int8"](grid, QUANTIZE_THREADS, x, mean_arg, x8, scale, n, block, token_layout(x),
                                     *rope_args(rope))
    return x8, scale


V8_PAD_KEYS = 64  # must match csrc/flash.cu


def quantize_v(v: torch.Tensor, layout: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """v ≈ v8 · scale with one scale per channel over all tokens of a head: int8 [batch, heads, N, 128] and
    float32 [batch, heads, 128]. d is the axis P·V does not sum over, so the kernel applies the scale once,
    at the end. Bit for bit bench/emulate.py's V quantization. layout 1 and 2 (for attention_int8pv_tma,
    csrc/flash.cu v8_offset) reorder the same bytes and pad each head to whole tiles: [batch, heads, N', 128]
    with N' = N rounded up to V8_PAD_KEYS."""
    v = readable(v)
    low, high = torch.aminmax(v, dim=-2)  # one pass, and |v| needs no copy of v
    scale = (torch.maximum(high, -low).float() / 127).clamp_min(1e-12)
    batch, heads, n, dim = v.shape
    tokens = n if layout == 0 else math.ceil(n / V8_PAD_KEYS) * V8_PAD_KEYS
    v8 = torch.empty(batch, heads, tokens, dim, dtype=torch.int8, device=v.device)
    n_chunks = v.numel() // CHUNK
    grid = math.ceil(n_chunks / QUANTIZE_THREADS)
    module("flash")["quantize_v"](grid, QUANTIZE_THREADS, v, scale, v8, n, n_chunks, layout, token_layout(v))
    return v8, scale


FP4_TILE = 64  # keys per tile of V4, the k of one m16n8k64 mma
V4_ROW_BYTES = 48  # these two must match csrc/flash.cu
E2M1_MAX, E4M3_MAX = 6.0, 448.0


def quantize_v4(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """V for attention_fp4pv (M4.8): NVFP4 per channel, e2m1 with an e4m3 scale per 16 keys, under one float
    scale per channel, max |v| / (6 · 448). Returns uint8 [batch, heads, tiles, 128, 48] (csrc/flash.cu V4: the
    keys of each tile of 64 in the mma's slot order, and their scales) and float32 [batch, heads, 128]. Bit for
    bit bench/emulate.py's block_scaled(..., "nvfp4", outer="channel") on V in that order."""
    v = readable(v)
    low, high = torch.aminmax(v, dim=-2)
    outer = (torch.maximum(high, -low).float() / (E2M1_MAX * E4M3_MAX)).clamp_min(1e-30)
    batch, heads, n, dim = v.shape
    tiles = math.ceil(n / FP4_TILE)
    v4 = torch.empty(batch, heads, tiles, dim, V4_ROW_BYTES, dtype=torch.uint8, device=v.device)
    n_threads = batch * heads * tiles * (FP4_TILE // 16) * dim
    module("flash")["quantize_v4"](math.ceil(n_threads / QUANTIZE_THREADS), QUANTIZE_THREADS, v, outer, v4, n,
                                   n_threads, token_layout(v))
    return v4, outer


FP4_ROW_BYTES = 80  # these three must match csrc/flash.cu
BIAS_THREADS, BIAS_KEYS = 128, 64


def quantize_fp4_rows(x: torch.Tensor, mean: torch.Tensor | None, pad_to: int = 1, rope: torch.Tensor | None = None,
                      low_high: tuple[torch.Tensor, torch.Tensor] | None = None,
                      mean_tokens: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Q or K for attention_fp4 (M4.12): x − mean in NVFP4 along d, e2m1 with an e4m3 scale per 16 channels, under one
    float scale per head, max |x − mean| / (6 · 448). Returns uint8 [batch, heads, N, 80] (csrc/flash.cu K4: 64 bytes of
    e2m1 in channel order, then the 8 scales) and float32 [batch, heads]. Bit for bit bench/emulate.py's
    block_scaled(x − mean, "nvfp4", -1). pad_to: rows per head rounded up to a multiple of it, the extra rows zeros
    (K for attention_fp4: whole tiles, so that a tile is one contiguous block). rope: x rotated first; low_high: the
    per-channel min and max of the rotated x (rope_stats), which the head's scale then comes from. mean_tokens (M4.21):
    mean is [batch, heads, blocks, 128], one mean per block of that many tokens, and low_high then per block too."""
    x = readable(x)
    batch, heads, n, dim = x.shape
    # max |x − mean| over the head from each channel's min and max (one pass, no float copy of x): the extreme
    # elements go through the same float subtraction as in the emulator, so the max is the same bit for bit
    low, high = low_high if low_high is not None else torch.aminmax(x, dim=-2)
    low, high = low.float(), high.float()
    if mean is not None:
        low, high = low - mean, high - mean
    amax = torch.maximum(high.abs(), low.abs()).flatten(2).amax(-1)
    outer = (amax / (E2M1_MAX * E4M3_MAX)).clamp_min(1e-30)
    tokens = math.ceil(n / pad_to) * pad_to
    x4 = torch.empty(batch, heads, tokens, FP4_ROW_BYTES, dtype=torch.uint8, device=x.device)
    x4[:, :, n:] = 0
    n_threads = batch * heads * n * (dim // 16)
    mean_arg = mean.contiguous() if mean is not None else ctypes.c_void_p(0)
    module("flash")["quantize_fp4_rows"](math.ceil(n_threads / QUANTIZE_THREADS), QUANTIZE_THREADS, x, mean_arg, outer,
                                         x4, n, tokens, n_threads, token_layout(x), *rope_args(rope), mean_tokens or n)
    return x4, outer


def smooth_bias(k: torch.Tensor, k_mean: torch.Tensor, q_mean: torch.Tensor, qk_outer: torch.Tensor, padded_keys: int,
                rope: torch.Tensor | None = None) -> torch.Tensor:
    """attention_fp4's Q smoothing (M4.21): q̄_B · (k − k̄) / qk_outer, the scores' share of Q's mean over each block B,
    which the kernel adds back. Near float, from the bf16 k (rotated by rope first): q̄ and k − k̄ each go in as two bf16
    (csrc/flash.cu). q_mean float32 [batch, heads, blocks, 128], k_mean [batch, heads, 128]. Returns float32 [batch,
    heads, blocks, padded_keys], 0 past the keys."""
    k = readable(k)
    batch, heads, n, _ = k.shape
    blocks = q_mean.shape[2]
    rows = torch.nn.functional.pad(q_mean, (0, 0, 0, -blocks % 16))  # whole mma rows of 16
    q_hi = rows.bfloat16()
    q_lo = (rows - q_hi.float()).bfloat16()
    row, pair = _a_fragments(rows.shape[2], k.device)
    q_hi, q_lo = (x.view(torch.int32)[..., row, pair].contiguous() for x in (q_hi, q_lo))
    bias = torch.empty(batch, heads, blocks, padded_keys, device=k.device)
    module("flash")["smooth_bias"]((math.ceil(padded_keys / BIAS_KEYS), batch * heads), BIAS_THREADS, k,
                                   k_mean.contiguous(), q_hi, q_lo, qk_outer, bias, n, padded_keys, blocks,
                                   rows.shape[2], token_layout(k), *rope_args(rope))
    return bias


def _a_fragments(n_rows: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Indices that put a [rows, 64 bf16 pairs] matrix into m16n8k16 A fragments, [rows / 16, 8 k-steps, 32 lanes, 4]:
    register r of lane l in k-step s holds row 16 · tile + l / 4 + 8 · (r % 2), pair 8s + l % 4 + 4 · (r / 2)."""
    tile, step, lane, reg = torch.meshgrid(*(torch.arange(n, device=device) for n in (n_rows // 16, 8, 32, 4)),
                                           indexing="ij")
    return 16 * tile + lane // 4 + 8 * (reg % 2), 8 * step + lane % 4 + 4 * (reg // 2)


ROPE_PAIRS = HEAD_DIM // 2
STATS_TOKENS = 256  # must match csrc/flash.cu


def rope_table(rope: torch.Tensor, batch: int, n_queries: int, n_keys: int) -> torch.Tensor:
    """ComfyUI's RoPE table checked: float32 [batch or 1, tokens, 1, 64, 2, 2], self-attention."""
    if n_queries != n_keys:
        raise ValueError(f"rope= is for self-attention, got {n_queries} queries and {n_keys} keys")
    if (rope.dtype != torch.float32 or rope.dim() != 6 or rope.shape[1] != n_queries or rope.shape[2] != 1
            or rope.shape[3:] != (ROPE_PAIRS, 2, 2) or rope.shape[0] not in (1, batch)):
        raise ValueError(f"rope must be float32 [{batch} or 1, {n_queries}, 1, {ROPE_PAIRS}, 2, 2], "
                         f"got {rope.dtype} {tuple(rope.shape)}")
    return rope


# the last table compacted: (a weak reference to it, its version, the compact table). The 40 blocks of a step share one
# table; a weak reference, not the address, because ComfyUI builds a new table every step and the allocator may give it
# the old one's address
_compact_rope: tuple = (lambda: None, -1, None)


def compact_rope(rope: torch.Tensor) -> torch.Tensor:
    """The table as the kernels take it (M4.19): (cos, sin) per pair in fp16, [batch or 1, tokens, 64, 2]. The table must
    be rotations, [[c, −s], [s, c]], as WAN's is. Converted once and kept while the same table comes back."""
    global _compact_rope
    # an inference tensor (ComfyUI samples under torch.inference_mode) has no version counter; the weak reference
    # still tells a new table from the kept one
    version = None if rope.is_inference() else rope._version
    ref, kept_version, compact = _compact_rope
    if ref() is rope and kept_version == version:
        return compact
    cos, sin = rope[..., 0, 0], rope[..., 1, 0]
    if not (torch.equal(rope[..., 1, 1], cos) and torch.equal(rope[..., 0, 1], -sin)):
        raise ValueError("rope must hold rotations [[cos, -sin], [sin, cos]], as WAN's table does")
    compact = torch.stack([cos, sin], -1)[:, :, 0].half().contiguous()
    _compact_rope = (weakref.ref(rope), version, compact)
    return compact


def rope_args(rope: torch.Tensor | None) -> tuple:
    """The (freqs, freqs_batch) kernel arguments for ComfyUI's table: the compact table and its batch stride in halves
    (0: shared), or null."""
    if rope is None:
        return ctypes.c_void_p(0), ctypes.c_longlong(0)
    compact = compact_rope(rope)
    return compact, ctypes.c_longlong(compact.stride(0) if compact.shape[0] > 1 else 0)


def rope_stats(x: torch.Tensor, rope: torch.Tensor | None,
               per: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per channel of each head, over all tokens of x rotated by rope (None: x as it is): mean, min and max, float32
    [batch, heads, 128]. One pass over x; the rotated values are never written. The partial sums of the blocks of
    STATS_TOKENS tokens are added in a fixed order, so the mean is the same from run to run. per (M4.21, a multiple of
    STATS_TOKENS): the same over each block of that many tokens instead, [batch, heads, blocks, 128]."""
    x = readable(x)
    batch, heads, n, dim = x.shape
    blocks = math.ceil(n / STATS_TOKENS)
    partial = torch.empty(batch * heads, blocks, 3, dim, device=x.device)
    module("flash")["rope_stats"]((blocks, batch * heads), QUANTIZE_THREADS, x, *rope_args(rope), partial, n,
                                  token_layout(x))
    partial = partial.view(batch, heads, blocks, 3, dim)
    sums, low, high = partial[..., 0, :], partial[..., 1, :], partial[..., 2, :]
    if per is None:
        return sums.sum(-2) / n, low.amin(-2), high.amax(-2)
    if per % STATS_TOKENS:
        raise ValueError(f"per must be a multiple of {STATS_TOKENS}, got {per}")
    group = per // STATS_TOKENS
    pad = (0, 0, 0, -blocks % group)  # the last block of `per` may hold fewer partials
    grouped = lambda t, value: torch.nn.functional.pad(t, pad, value=value).unflatten(-2, (-1, group))  # noqa: E731
    counts = torch.full((math.ceil(n / per), 1), float(per), device=x.device)
    counts[-1] = n - per * (counts.shape[0] - 1)
    return grouped(sums, 0.0).sum(-2) / counts, grouped(low, math.inf).amin(-2), grouped(high, -math.inf).amax(-2)
