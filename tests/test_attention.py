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
    "attention_int8pv": dict(INT8_QK, p_fmt="u8", v_fmt="int8", p_local=True, sum_rounded=True),
}
EMULATED["attention_int8pv_async"] = EMULATED["attention_int8pv_pipe"] = EMULATED["attention_int8pv"]
EMULATED["attention_int8pv_tma"] = EMULATED["attention_int8pv"]
# M4.8: P against the running max times 2^10 and V per channel, both NVFP4, keys in the FP4 mma's slot order
EMULATED["attention_fp4pv"] = dict(EMULATED["attention_int8pv"], p_fmt="nvfp4", v_fmt="nvfp4", p_local=False,
                                   p_mul=1024.0, p_outer=1.0, v_outer="channel", fp4_slots=True)
EMULATED["attention_fp4"] = dict(EMULATED["attention_fp4pv"], qk="nvfp4")  # M4.12: Q·Kᵀ in NVFP4 too
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
# M3.7: more blocks per SM for the u8 kernel; a 32-key tile is what lets 3 or 4 blocks fit in shared memory
ASYNC_VARIANTS = [
    {"MIN_BLOCKS_PER_SM": 3},
    {"KEYS_PER_TILE": 32},
    {"KEYS_PER_TILE": 32, "MIN_BLOCKS_PER_SM": 3},
    {"KEYS_PER_TILE": 32, "MIN_BLOCKS_PER_SM": 4},
]


def test_every_kernel_matches_fp32_reference_on_all_shapes():
    torch.manual_seed(0)
    for batch, heads, n_queries, n_keys in SHAPES:
        q = torch.randn(batch, heads, n_queries, 128, device="cuda").bfloat16()
        k, v = (torch.randn(batch, heads, n_keys, 128, device="cuda").bfloat16() for _ in range(2))
        ref = reference(q, k, v)
        floor = errors(ref.bfloat16().float(), ref)["rel_l2"]

        variants = [(kernel, {}) for kernel in unica.KERNELS] + [("attention_smem", t) for t in TILE_VARIANTS]
        variants += [("attention_int8pv_async", t) for t in ASYNC_VARIANTS]
        for kernel, tile in variants:
            out = unica.attention(q, k, v, kernel=kernel, **tile)
            error = errors(out.float(), ref)["rel_l2"]
            name = kernel + "".join(f" {key}={value}" for key, value in tile.items())
            assert torch.isfinite(out).all(), name
            if kernel in unica.QUANTIZED:
                # each tile has its own P max in the u8 kernels, so the emulator needs the same tiles
                emulated = emulate(q, k, v, **EMULATED[kernel], bc=tile.get("KEYS_PER_TILE", unica.TILE["KEYS_PER_TILE"]))
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


def test_pipelined_qk_changes_nothing_but_time():
    torch.manual_seed(7)
    # 1, 2 and 3 tiles: the prologue, the last iteration's thrown-away Q·Kᵀ, the K(t + 2) loads
    for batch, heads, n_queries, n_keys in SHAPES + [(1, 2, 100, 128), (1, 2, 100, 129), (1, 2, 100, 192)]:
        q = torch.randn(batch, heads, n_queries, 128, device="cuda").bfloat16()
        k, v = (torch.randn(batch, heads, n_keys, 128, device="cuda").bfloat16() for _ in range(2))
        inputs = unica.kernel_inputs(q, k, v, "attention_int8pv_async")
        for tile in ({}, {"KEYS_PER_TILE": 32}):
            in_turn = unica.attention(q, k, v, "attention_int8pv_async", inputs, **tile)
            pipelined = unica.attention(q, k, v, "attention_int8pv_pipe", inputs, **tile)
            assert torch.equal(pipelined, in_turn), (tile, n_queries, n_keys)


def test_tma_tiles_change_nothing_but_time():
    torch.manual_seed(9)
    # 1, 2 and 3 tiles, partial ones, several heads (a K box past one head reads the next), cross-attention
    for batch, heads, n_queries, n_keys in SHAPES + [(1, 2, 100, 128), (1, 2, 100, 129), (1, 2, 100, 192)]:
        q = torch.randn(batch, heads, n_queries, 128, device="cuda").bfloat16()
        k, v = (torch.randn(batch, heads, n_keys, 128, device="cuda").bfloat16() for _ in range(2))
        for tile in ({}, {"KEYS_PER_TILE": 32}):
            copied = unica.attention(q, k, v, "attention_int8pv_pipe", **tile)
            for pairs in (0, 1):
                by_tma = unica.attention(q, k, v, "attention_int8pv_tma", V_PAIRS=pairs, **tile)
                assert torch.equal(by_tma, copied), (tile, pairs, n_queries, n_keys)


def test_bare_exp2_changes_nothing_but_time():
    torch.manual_seed(2)
    for batch, heads, n_queries, n_keys in SHAPES:
        q = torch.randn(batch, heads, n_queries, 128, device="cuda").bfloat16()
        k, v = (torch.randn(batch, heads, n_keys, 128, device="cuda").bfloat16() for _ in range(2))
        inputs = unica.kernel_inputs(q, k, v, "attention_int8pv_async")
        exp2f = unica.attention(q, k, v, "attention_int8pv_async", inputs, FAST_EXP2=0)
        assert torch.equal(unica.attention(q, k, v, "attention_int8pv_async", inputs), exp2f), (n_queries, n_keys)


def test_lazy_rescale_changes_nothing_but_time():
    torch.manual_seed(4)
    for batch, heads, n_queries, n_keys in SHAPES:
        q = torch.randn(batch, heads, n_queries, 128, device="cuda").bfloat16()
        k, v = (torch.randn(batch, heads, n_keys, 128, device="cuda") for _ in range(2))
        # keys that grow along the sequence: a new max in nearly every tile, so the rescale branch runs too
        growing = k * torch.linspace(0.5, 3.0, n_keys, device="cuda")[:, None]
        for keys in (k, growing):
            keys = keys.bfloat16()
            for kernel in ("attention_int8pv", "attention_int8pv_async"):
                inputs = unica.kernel_inputs(q, keys, v.bfloat16(), kernel)
                every_tile = unica.attention(q, keys, v.bfloat16(), kernel, inputs, LAZY_RESCALE=0)
                lazy = unica.attention(q, keys, v.bfloat16(), kernel, inputs)
                assert torch.equal(lazy, every_tile), (kernel, n_queries, n_keys)


def test_both_denominators_match_their_emulator():
    # random and sharp attention (q · 4): Σ P̂ and Σ P part on sharp data, so each build must land on its own
    for sharp in (1.0, 4.0):
        torch.manual_seed(8)
        q = (torch.randn(1, 2, 1000, 128, device="cuda") * sharp).bfloat16()
        k, v = (torch.randn(1, 2, 1000, 128, device="cuda").bfloat16() for _ in range(2))
        for rounded in (0, 1):
            emulated = emulate(q, k, v, **dict(EMULATED["attention_int8pv"], sum_rounded=bool(rounded)))
            for kernel in ("attention_int8pv_async", "attention_int8pv_pipe"):
                out = unica.attention(q, k, v, kernel, SUM_ROUNDED=rounded)
                from_emulator = errors(out.float(), emulated)["rel_l2"]
                assert from_emulator < MAX_FROM_EMULATOR, (kernel, sharp, rounded, from_emulator)


def test_skipped_tiles_are_the_emulators():
    # M4.44: random attention has no negligible tile, so SKIP_TILES (a warp alone) and SKIP_BLOCKS (the block of 64
    # rows) leave it bit for bit. Sharp attention whose rows are alike in groups of 16 or 64 (q · 8, one query per
    # group plus noise) has many: the kernel skips them as the emulator does.
    import emulate as emulator
    torch.manual_seed(9)
    q, k, v = (torch.randn(1, 2, 1000, 128, device="cuda") for _ in range(3))
    q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    inputs = unica.kernel_inputs(q, k, v, "attention_int8pv_tma")
    plain = unica.attention(q, k, v, "attention_int8pv_tma", inputs)
    for setting, rows in (("SKIP_TILES", 16), ("SKIP_BLOCKS", 64)):
        assert torch.equal(unica.attention(q, k, v, "attention_int8pv_tma", inputs, **{setting: 10}), plain), setting

        sharp = ((q.float()[:, :, ::rows].repeat_interleave(rows, 2)[:, :, :1000] + 0.1 * q.float()) * 8).bfloat16()
        stats, block_rows = {}, emulator.SKIP_ROWS
        emulator.SKIP_ROWS = rows
        try:
            emulated = emulate(sharp, k, v, **EMULATED["attention_int8pv_tma"], skip_below=2**-10, stats=stats)
        finally:
            emulator.SKIP_ROWS = block_rows
        assert stats["skipped"] > 0.1 * stats["pairs"], (setting, stats)
        out = unica.attention(sharp, k, v, "attention_int8pv_tma", **{setting: 10})
        assert errors(out.float(), emulated)["rel_l2"] < MAX_FROM_EMULATOR, setting


def test_producer_warps_change_nothing_but_time():
    # M4.48: a producer warpgroup issues the copies and the computing warps stop waiting for each other: the same
    # arithmetic, so bit for bit, for each V scale, 1 to many tiles, partial ones, cross-attention. Three runs each,
    # since a wrong barrier phase shows only in some.
    torch.manual_seed(11)
    for batch, heads, n_queries, n_keys in SHAPES + [(1, 2, 100, 64), (1, 2, 100, 129), (1, 2, 1000, 1000)]:
        q = torch.randn(batch, heads, n_queries, 128, device="cuda").bfloat16()
        k, v = (torch.randn(batch, heads, n_keys, 128, device="cuda").bfloat16() for _ in range(2))
        for scale in ({}, {"V_TOKEN": 1}):
            inputs = unica.kernel_inputs(q, k, v, "attention_int8pv_tma", **scale)
            plain = unica.attention(q, k, v, "attention_int8pv_tma", inputs, **scale)
            for setting in ({"PRODUCER": 1}, {"RECOMPUTE": 0}, {"PRODUCER": 1, "RECOMPUTE": 0},  # M4.49 too
                            {"PRODUCER": 1, "WAIT_HINT": 1000},  # and M4.51
                            {"PRODUCER": 1, "WARPS_PER_BLOCK": 8}):  # and M4.53: two warpgroups share K and V
                for _ in range(3):
                    other = unica.attention(q, k, v, "attention_int8pv_tma", inputs, **setting, **scale)
                    assert torch.equal(other, plain), (setting, scale, n_queries, n_keys)


def test_wave_dispatch_changes_nothing_but_time():
    # M4.55: the presets choose 8 computing warps per call where WAVE_RULE says so (bit for bit the same output, M4.53),
    # and 4 where the last wave or a short loop would cost
    device = torch.device("cuda")
    sms = unica.sm_count(device)
    assert unica.wave_layout(sms, 128 * 128, 16384, 0, device) == unica.EIGHT_WARPS  # 128 waves exactly, long enough
    assert unica.wave_layout(sms, 128 * 32, 4096, 0, device) == {}  # too short
    assert unica.wave_layout(1, 128 * 128, 16384, 0, device) == {}  # 128 blocks: 2 waves, the second 44 / 84 full
    torch.manual_seed(12)
    for preset, n in (("per_channel", 16384), ("per_token", 16384)):
        heads = sms // 4  # with 128-row blocks: n / 128 · heads / sms whole waves
        q, k, v = (torch.randn(1, heads, n, 128, device="cuda").bfloat16() for _ in range(3))
        assert unica.wave_layout(heads, n, n, unica.PRESETS[preset].get("V_TOKEN", 0), device), preset
        settings = unica.PRESETS[preset]
        assert torch.equal(unica.attention(q, k, v, **settings), unica.attention(q, k, v, **{**settings, "WAVE_DISPATCH": 0}))


def test_trace_build_changes_nothing_but_time():
    torch.manual_seed(3)
    q, k, v = (torch.randn(1, 2, 1000, 128, device="cuda").bfloat16() for _ in range(3))
    inputs = unica.kernel_inputs(q, k, v, "attention_int8pv_async")
    plain = unica.attention(q, k, v, "attention_int8pv_async", inputs)
    assert torch.equal(unica.attention(q, k, v, "attention_int8pv_async", inputs, TRACE=1), plain)


def test_rejects_what_the_kernel_cannot_take():
    q = torch.randn(1, 1, 16, 64, device="cuda").bfloat16()  # head_dim 64
    for bad in (q, q.float()):
        try:
            unica.attention(bad, bad, bad)
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"accepted {bad.dtype} {tuple(bad.shape)}")


def test_callers_layout_changes_nothing_but_copies():
    """M4.17: q, k, v in ComfyUI's [batch, tokens, heads, 128], seen through a transpose, give bit for bit the output
    of the same values made contiguous, and the output comes back in that layout."""
    torch.manual_seed(2)
    for batch, heads, n_queries, n_keys in SHAPES:
        q = torch.randn(batch, n_queries, heads, 128, device="cuda").bfloat16()
        k, v = (torch.randn(batch, n_keys, heads, 128, device="cuda").bfloat16() for _ in range(2))
        for kernel in sorted(unica.LAYOUT_KERNELS):
            strided = unica.attention(*(x.transpose(1, 2) for x in (q, k, v)), kernel)
            dense = unica.attention(*(x.transpose(1, 2).contiguous() for x in (q, k, v)), kernel)
            assert torch.equal(strided, dense), (kernel, batch, heads, n_queries, n_keys)
            assert strided.transpose(1, 2).is_contiguous() or n_queries == 1, (kernel, strided.stride())


if __name__ == "__main__":
    test_callers_layout_changes_nothing_but_copies()
    test_every_kernel_matches_fp32_reference_on_all_shapes()
    test_async_loads_change_nothing_but_time()
    test_pipelined_qk_changes_nothing_but_time()
    test_tma_tiles_change_nothing_but_time()
    test_bare_exp2_changes_nothing_but_time()
    test_lazy_rescale_changes_nothing_but_time()
    test_both_denominators_match_their_emulator()
    test_skipped_tiles_are_the_emulators()
    test_producer_warps_change_nothing_but_time()
    test_wave_dispatch_changes_nothing_but_time()
    test_trace_build_changes_nothing_but_time()
    test_rejects_what_the_kernel_cannot_take()
    print("attention ok")
