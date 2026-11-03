# UnicaAttention — optimization log

Every significant change: the hypothesis written before measuring, the measurement, the verdict.
Kept whether the hypothesis held or not. Times come from `bench/compare.py`, variants side by side
in one run (run-to-run drift is ~10 %). Shape 1 × 40 × 32760 × 128, bf16, random inputs.

## M1 — baseline

`attention`: plain flash attention on `mma.sync`, every warp loads its own fragments from global
memory, 4-byte loads (2-byte for V). 695–771 ms (0.7× SDPA), rel L2 0.17 % (bf16 output floor + 0.001 %).

Nsight Compute (4 heads): latency-bound. Tensor pipe active 27.5 %, issue slots busy 8 %, 24 cycles
between two instructions of a warp, 76 % of them waiting on L1 (long scoreboard). VRAM 1.6 % busy,
L1 hit 94 %. 184 registers → 2 blocks = 8 warps per SM.

## M2.1 — K and V tiles in shared memory, fragments by ldmatrix

**Attacks:** the number of loads each warp waits on. Per tile and warp: 128 4-byte loads of K and
256 2-byte loads of V become 16 16-byte loads (shared by the block) and 64 `ldmatrix.x4`
(`.trans` for V). Rows padded to 272 bytes so the 8 rows of one ldmatrix fall in different banks.

**Hypothesis:** ~2× faster, 770 → ~380 ms (300–450), because loads drop ~5× but the block still
waits for each tile at `__syncthreads`, with only the second block on the SM to cover it.

**Measured:** 695.4 → **213.9 ms, 3.25×**, 103 TFLOPS, rel L2 unchanged (0.17 %).

**Verdict: the hypothesis was wrong — too pessimistic.** The barrier wait barely matters: with 2
blocks per SM one computes while the other loads (barrier stalls are 5 % of math-throttle stalls).
Once the loads are few and wide, the limit moves to the tensor cores themselves.

| Nsight Compute (4 heads) | `attention` | `attention_smem` |
|---|---|---|
| tensor pipe (HMMA) active | 27.5 % | **87.7 %** |
| issue slots busy | 8.3 % | 24.4 % |
| cycles between two instructions of a warp | 24.1 | 8.1 |
| top stall | L1 wait (long scoreboard) | tensor pipe busy (math pipe throttle) |
| shared-memory bank conflicts | — | 11.2 M (~3 per warp and tile, small; not yet explained) |

Machine code: 184 → 236 registers, no spills; per tile LDG 400 → 16, LDSM 0 → 64, STS 16, BAR 2.
Occupancy unchanged: 2 blocks = 8 warps per SM, now limited by registers and shared memory (34 KB) alike.

**How close to the tensor-core peak.** A first estimate said 93 %, but it divided 103 TFLOPS from one
run by 111 from another: the drift this log warns about. Measured in one process (`mma_peak` in
csrc/flash.cu, the same half-bf16 half-fp16 m16n8k16 mix with FP32 accumulate, alternating rounds):
peak 112.7 TFLOPS, `attention_smem` 92.0 TFLOPS of executed mma = **81.6 % of peak**. Executed mma
equal the useful ones within 0.05 % (32760 padded to 32768). Pipe-active 87.5 % (ncu) is not
throughput: it counts cycles the pipe is busy, not mma per cycle.

## M2.2 — tile shape

Built from the same source with `-D UNICA_*` (`unica.attention(..., KEYS_PER_TILE=32)`), all shapes
correct on every test shape first. W = warps per block (16 query rows each), K = keys per tile,
B3 = `__launch_bounds__(128, 3)`, ptxas fits registers for 3 blocks per SM.

**Hypothesis:** the current W4 K64 is the fastest; tensor-bound, so the others lose 0–15 %.

| Shape | Registers, spills | Shared | Blocks / warps per SM | Predicted | Measured (same run) | Verdict |
|---|---|---|---|---|---|---|
| **W4 K64** | 242, none | 34.8 KB | 2 / 8 | reference | **225.0 ms** | fastest, as predicted |
| W8 K64 | 204, none | 34.8 KB | 1 / 8 | +5–10 %: one block, nobody covers its barrier | +6.8 % | held: barrier stalls +43 %, tensor pipe 86 → 80 % |
| W4 K32 | 188, none | 17.4 KB | 2 / 8 | +10 %: twice the barriers, shuffles and rescales per key | +2.8 % | wrong: per-tile overhead is cheaper than estimated |
| W8 K32 | 182, none | 17.4 KB | 1 / 8 | +10–15 % | +17.7 % | direction held, size underestimated |
| W4 K64 B3 | 168, **none** | 34.8 KB | **2** / 8 | +10 % from spills | +9.6 % | right number, **wrong reason**: no spills; 3 × (34.8 + 1 reserved) KB = 107 KB > 100 KB, so shared memory still allows only 2 blocks. Registers were squeezed for nothing: L1 waits (long scoreboard) +69 % |
| W4 K32 B3 | 168, none | 17.4 KB | **3 / 11.7** | ~215 ms, −4 %: 12 warps keep the tensor pipe fed | +1.7 % (226.3 vs 222.6 in its own run, within the 4–11 % spread) | wrong: tensor pipe 86 → 88.5 %, but K32's per-tile overhead eats the gain |

**Result:** W4 K64 stays. Near the tensor-core limit the shape moves time by only a few percent;
what costs is one block per SM (the W8 shapes lose 7–18 %). The register count is not the lever
here, shared memory decides how many blocks fit as often as registers do.

## M3.1 — INT8 Q·Kᵀ

`attention_int8qk`: Q and K int8 with one scale per 128 Q tokens and per 64 K tokens (one per tile),
K smoothed (mean over tokens subtracted). `mma.sync.m16n8k32.s32.s8.s8.s32`, exact int32 accumulate.
The tile shares one Q scale and one K scale, so dequantization is a single number in the place of
`scale_log2`: no instruction of its own, only 32 `I2F` per tile. P·V unchanged (FP16, from shared
memory). Quantization runs in torch before the kernel, for now.

**Hypotheses:** kernel 225 → ~155 ms (140–175): Q·Kᵀ is half the tensor work and gets 4× cheaper,
tensor floor 200 → 125 ms, at ~82 % efficiency ~155 ms. Torch quantization +5–10 ms. Error as the
emulator predicts, ~1.3 %.

**Measured** (same run): kernel alone 218.5 → **149.9 ms (1.46×)**; with torch quantization 168.9 ms;
quantization alone **21.7 ms**. Error 1.32 % at full shape; on every test shape the emulator's
prediction to the third digit (1.231 % / 1.231 %, 1.337 % / 1.337 %), kernel vs emulator output 0.08 %.

**Verdict:** kernel hypothesis held. Quantization hypothesis wrong, 2–4× over: torch makes about ten
float32 passes over 670 MB. The traffic it needs is ~1.3 GB (read Q, K twice for the mean, write int8),
~1.7 ms at 810 GB/s: a CUDA quantization kernel should save ~20 ms end to end.

Machine code: 242 → 230 registers, no spills; per tile 128 HMMA → 64 HMMA + 32 IMMA, LDSM 64 → 48,
+32 I2F; shared memory 34.8 → 26.6 KB (3 blocks would now fit by shared memory, registers still allow 2).

| Nsight Compute (4 heads) | `attention_smem` | `attention_int8qk` |
|---|---|---|
| tensor pipe, HMMA (now P·V only) | 87.7 % | 65.7 % |
| tensor pipe, IMMA (Q·Kᵀ) | — | 15.2 % |
| issue slots busy | 24.4 % | 33.9 % |
| stalls: math pipe throttle | 2.21 M | 0.99 M |
| stalls: long scoreboard (global → shared tile loads) | 0.61 M | 0.66 M |

P·V is now two thirds of the tensor time, and the tile loads (`LDG` → `STS`, through registers) start
to show: the place for FP8 P·V and for `cp.async` / TMA.

## M3.2 — quantization in CUDA

`quantize_int8` in csrc/flash.cu: one block of 256 threads per scale block, max over the block, then
a second read (from L2) that writes int8. The K mean stays in torch (`k.mean(-2, dtype=float32)`, one
pass, 0.39 ms). Bit for bit bench/emulate.py's `quant()` (tests/test_quantize.py), which needed one
finding: torch divides a tensor by a scalar as a multiplication by the reciprocal, so the scale is
`max · (1/127)`, not `max / 127`; the two differed in the last bit for 1 block in 32.

**Hypothesis:** 21.7 → ~2 ms (2–3): ~1.34 GB of VRAM traffic at 810 GB/s is 1.65 ms.

**Measured** (same run): torch 22.6 ms → **CUDA 1.59 ms (14×)**, 0.39 ms of it the torch K mean.
Nsight Compute: DRAM throughput **90 %** in both launches (Q 573 µs, K 554 µs): bound by VRAM, as it
should be. End to end, `attention_int8qk` / `attention_smem` = 0.686 in the same run, the ratio the
kernels alone had (0.686): the quantization no longer shows.

**Verdict:** held. This run was ~14 % slower overall than the previous ones (`attention_smem`
249.6 ms vs 218.5 ms), all kernels together; only the ratios are comparable across runs.

## M3.3 — P·V format: u8 × s8 or e4m3 × e4m3 (prediction, before implementation)

**Measured facts** (bench/peak.py, csrc/peak.cu, all forms alternating in one run, 8 independent chains
per warp, 16 warps per SM):

| Form | SASS | Rate |
|---|---|---|
| m16n8k16 bf16/f16 → f32 (today's exact path) | HMMA.16816.F32(.BF16) | 114.1 TFLOPS |
| m16n8k32 s8 × s8 → s32 (Q·Kᵀ) | IMMA.16832.S8.S8 | 453.6 TOPS |
| m16n8k32 u8 × s8 → s32 | IMMA.16832.U8.S8 | 465.2 TOPS |
| m16n8k32 e4m3 × e4m3 → f16 | QMMA.16832.F16.E4M3.E4M3 | 456.1 TFLOPS |
| m16n8k32 e4m3 × e4m3 → f32 | QMMA.16832.F32.E4M3.E4M3 | **226.5 TFLOPS (half rate)** |
| s8 × s8 alternating with u8 × s8 (candidate A) | | 450.9 |
| s8 × s8 alternating with e4m3 → f16 (candidate B) | | 448.5 |
| int32 → f32 (`I2FP.F32.S32`) | ALU pipe, not the SFU (ncu: xu = 0) | ≥ 8.8 T/s (issue-bound with its IADD and FADD) |
| `ex2.approx` | MUFU.EX2 | 3.9 T/s |

So the two mixes differ by 0.5 %: equal tensor rate, measured, as long as FP8 accumulates in f16.
Also on sm_120a, compiled: `ldmatrix.m16n16.x1.trans.b8` (LDSM.8.MT1616, 8-bit V transposed on the way
in), `cvt.pack.sat.u8.s32.b32` (I2IP.U8.S32.SAT, packs u8), `cvt.rn.satfinite.e4m3x2.f32`
(F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C, packs e4m3 into a register half without a PRMT).

**Error, emulator** (8 × 32760, with INT8 Q·Kᵀ): u8 P (tile-local max) × s8 V per channel **1.75 %**,
e4m3 P × e4m3 V per channel **3.95 %** (f16 accumulate 3.96 %). The sources add in quadrature:
√(1.33² + 0.59² + 1.01²) = 1.77 and √(1.33² + 2.61² + 2.66²) = 3.96 (Q·Kᵀ, P, V alone).

**Prediction** (per warp and tile, against `attention_int8qk`; P·V 2 × 16 = 32 mma in both):

| | u8 × s8 | e4m3 × e4m3 → f16 |
|---|---|---|
| P·V mma | 32 IMMA.16832.U8.S8 | 32 QMMA.16832.F16.E4M3.E4M3 |
| quantizing P | ~48: 32 F2I + 16 I2IP (the ×255 folds into the exponent) | ~16 F2FP (the ×2 folds into the exponent) |
| flush per tile | +128: 64 I2FP + 64 FFMA (out = out·α + β·acc) | +128: 64 f16 → f32 + 64 FFMA |
| removed | 512 bf16 → f16 conversions of V, half the V ldmatrix | the same |
| registers | 210–235, no spills (accumulate one pair of d-blocks over both k-steps, then flush: 8 live accumulator registers, not 64) | ~4 fewer |
| kernel time | 0.42–0.55 × `attention_int8qk` | the same, 0–3 % faster (32 fewer instructions per tile) |
| error | 1.75 % ± 0.1 | 3.95 % ± 0.1 |

The tensor floor falls to ~32 % of today's time (P·V 4× cheaper), issue to ~23 %, the SFU stays at ~8 %.

**Decision:** u8 × s8 only. FP8 P·V at 3.95 % is not "well below SageAttention 2's 3.9 %" by
construction, and its rate is already measured above; it is not built.

## M3.3 — result: `attention_int8pv`, P·V in u8 × s8

Built as predicted, with two layout facts measured first on the card: `ldmatrix.m16n16.x2.trans.b8`
gives a lane keys 4·(lane % 4) .. + 3 of dim lane / 4 (and + 8), the m16n8k32 B fragment as is; and P
leaves Q·Kᵀ with keys 2t, 2t+1, 8+2t, 9+2t in lane t's bytes, which mma reads as k = 4t .. 4t+3. The sum
over k ignores order, so V's rows go into shared memory in P's order (`p_order_slot`): free, an address
change in the tile load. V is quantized once per head (`quantize_v`, max per channel by `torch.aminmax`).

Same run (bench/compare.py, 40 × 32760, 3 rounds, median of 5; the card ran slow, `attention_smem` 251.5
instead of ~225 ms):

| | `attention_int8qk` | `attention_int8pv` | predicted |
|---|---|---|---|
| whole call | 173.4 ms | **104.4 ms** | |
| quantization (Q, K / + V) | 1.65 ms | 2.81 ms | V ~1 ms: held (1.16) |
| kernel alone | 172.0 ms | **101.2 ms = 0.59×** | 0.42–0.55×: **wrong** |
| rel L2 vs FP32 | 1.32 % | **1.73 %** | 1.75 ± 0.1 %: held |
| kernel vs emulator | 0.08 % | **0.000–0.015 %** | ~0.1 %: better, P is now rounded alike in both |

Against SageAttention, one more run (whole calls, 3 rounds): SDPA 519.9, SageAttention 2 **75.1 ms,
3.90 %**, SageAttention 3 59.0 ms, 19.1 %, `attention_int8pv` **100.5 ms, 1.73 %**.

Machine code (ptxas -v, SASS of the main loop, per warp and tile):

| | `attention_int8qk` | `attention_int8pv` | predicted |
|---|---|---|---|
| registers, spills | 230, 0 | **253**, 0 | 210–235: wrong, occupancy unchanged (2 blocks / SM) |
| shared memory | 26.6 KB | 18.4 KB | |
| instructions | 1141 | **729** (0.64×) | |
| mma | 64 HMMA + 32 IMMA | 32 IMMA.S8.S8 + 32 IMMA.U8.S8 | held |
| quantizing P | 144 F2FP.F16 (fp16 P) | **16 F2IP.U8.F32.NTZ** | ~48: better, ptxas fuses rounding and packing, 2 values each |
| V conversion | 128 PRMT + 256 IMAD.U32 | 0 | held |
| flush | 64 FMUL | 64 FMUL + 64 I2FP + 64 FFMA | held |
| LDSM | 32 MT88.4 + 16 M88.4 | 16 MT1616.2 (8-bit) + 16 M88.4 | |
| exp2f | 34 MUFU.EX2 + 34 FSETP + ~68 FMUL | the same | not touched |

F2IP rounds halves to even, like torch.round (tests/test_mma.py checks exact halves).

Nsight Compute (4 heads):

| | `attention_int8qk` | `attention_int8pv` |
|---|---|---|
| duration | 21.11 ms | 12.08 ms |
| tensor pipe active | 80.0 % (HMMA 65.5 + IMMA 15.1) | **52.0 %** (IMMA) |
| issue slots busy | 31.5 % | 33.8 % |
| no eligible warp | 66.2 % | 62.9 % |
| shared-memory bank conflicts (loads) | 93.8 K | **0** |
| top stalls (M cycles) | math pipe 1.01, wait 0.74, long scoreboard 0.65 | wait 0.41, math pipe 0.38, long scoreboard 0.33, barrier 0.15 |

**Why the time prediction missed:** it assumed the tensor and issue floors would overlap as well as
before. They no longer bind: the tensor pipe is half idle and issue a third busy. With 2 warps per
scheduler (253 registers allow 2 blocks of 4 warps per SM), 63 % of cycles no warp is ready: the kernel
is now **latency-bound**. The waits are fixed-latency chains (exp2f's range check, the flush right after
its mma), the global → register → shared tile loads (long scoreboard) and the barrier around them.
What attacks this next: asynchronous tile loads (`cp.async` / TMA, the agreed step C), fewer
instructions per exponent (`ex2.approx.ftz` instead of `exp2f`: ~100 per tile), or more warps per SM.

## M3.4 — asynchronous tile loads (prediction, before implementation)

**What it attacks: latency, not bandwidth.** DRAM is 2.5 % busy and L2 18 %. Each tile today goes
global → registers → shared memory with two `__syncthreads` around it: every warp of the block waits for the
slowest load of the tile, an L2 round trip of several hundred cycles, and only the other block on the SM
works in the meantime. ncu: long scoreboard 0.33 M and barrier 0.15 M of ~1.97 M stall cycles, about a
quarter.

**Change** (`attention_int8pv_async`, otherwise `attention_int8pv` unchanged): `cp.async` copies 16 bytes
global → shared without passing through registers, and the thread does not wait for it. Two buffers: tile
t + 1 is loading while tile t is computed. One `__syncthreads` per tile instead of two: at the top of tile t
it says both "tile t has landed" and "every warp is done with tile t − 1", whose buffer the next load
reuses. Keys past the end: `cp.async` with source size 0 writes zeros. Why `cp.async` before TMA: the gain
here is the overlap, which both give. TMA also saves issue slots (one thread per tile), but needs a tensor
map from the driver and swizzled tiles instead of padded rows, which changes the ldmatrix addressing. It
is a separate step, measured on its own.

**Hypotheses:**

| | `attention_int8pv` | `attention_int8pv_async`, predicted |
|---|---|---|
| kernel time | 101 ms (0.59 × int8qk) | **−8 … −20 %: 80–93 ms**, most of the load and barrier stalls hidden |
| output | | **bit for bit the same**: the arithmetic does not change |
| registers | 253 | 235–253, no spills (the 16 registers that carried each uint4 are gone); still 2 blocks per SM |
| shared memory | 18.4 KB | 36.9 KB (two buffers); 2 blocks still fit in 100 KB |
| SASS per tile | 8 LDG.E.128 + 8 STS.128, 2 BAR.SYNC | 8 LDGSTS.E.BYPASS.128, LDGDEPBAR, DEPBAR, 1 BAR.SYNC |
| ncu | long scoreboard 0.33 M, barrier 0.15 M, tensor pipe 52 % | long scoreboard → ~0, barrier lower, tensor pipe ~58–65 % |

(The bank conflicts below were read only from `op_ld`, which hid the 11 M store conflicts of `int8pv`.
M3.6 has all the counters and the cause.)

## M3.4 — result

Same run (bench/compare.py, 40 × 32760, 5 rounds, median of 5):

| | `attention_int8pv` | `attention_int8pv_async` | predicted |
|---|---|---|---|
| kernel alone | 97.8 ms | **87.9 ms (−10.1 %)** | −8 … −20 %, 80–93 ms: **held, at the low end** |
| whole call | 101.0 ms | **90.2 ms** | |
| output | | identical bit for bit (test on all 6 shapes) | held |
| registers, spills | 253, 0 | 253, 0 | held; 2 blocks per SM, limited by registers |
| shared memory | 18.4 KB | 36.9 KB | held; the shared-memory limit falls from 5 blocks to 2, registers already allow only 2 |
| loop SASS per tile | 729: 8 LDG.E.128 + 8 STS.128, 2 BAR.SYNC | 734: 8 LDGSTS.E.BYPASS.128, LDGDEPBAR, DEPBAR.LE, 1 BAR.SYNC | held |

Nsight Compute (4 heads, its fixed clocks):

| | `attention_int8pv` | `attention_int8pv_async` |
|---|---|---|
| duration | 10.50 ms | 7.81 ms (−26 %) |
| tensor pipe active | 51.9 % | **62.0 %** |
| no eligible warp | 63.0 % | 55.5 % |
| stalls, long scoreboard / barrier (M) | 0.34 / 0.15 | **0.10 / 0.08** |
| stalls, short scoreboard (M) | 0.05 | 0.10 |
| shared-memory bank conflicts (`op_ld`) | 0 | 236 K |

**Not predicted:** the gain depends on the shape and on the clock: 40 heads −10 %, 4 heads −15 %
(CUDA events, 9.4 → 8.0 ms), ncu at its fixed clocks −26 %. Not explained yet. Also new: 236 K bank
conflicts counted as shared loads, and short-scoreboard stalls doubled. The ldmatrix addresses did not
change, so `LDGSTS` writing into permuted rows of the V tile is the first suspect; not checked yet.

Still latency-bound: in 55 % of cycles no warp is ready, and the top stalls are now fixed-latency waits
(0.39 M), math pipe (0.32 M) and the selected warp itself (0.38 M), with 2 warps per scheduler.

## M3.5 — the bare SFU exponent (prediction, before implementation)

**What it attacks: instructions in the softmax, the serial part of a tile.** Without fast math, `exp2f`
compiles to the same `MUFU.EX2` plus a range check that keeps results below 2^-126 as subnormals:
`FSETP.GEU x, -126`, `@!p FMUL x, 0.5` before it and `@!p FMUL r, r, r` after it. SASS of
`attention_int8pv_async`: 36 `MUFU.EX2` per tile (32 P values, alpha and beta of two rows), each with the
three extra instructions: **108 of the 734 loop instructions (15 %)**. The predicated FMULs do not run on
our data, but they still take an issue slot each, and the FSETP sits on the path to every MUFU.

**Change:** `ex2.approx.ftz.f32` directly in `online_softmax_u8` (both u8 kernels), behind
`UNICA_FAST_EXP2` (default 1) so that the old code stays measurable in the same run. Results below 2^-126
become 0: a P that small rounds to 0 in u8 anyway, and an alpha or beta that small multiplies a sum that
the new tile already dwarfs by 2^126. The fp16 kernels keep `exp2f`: not part of this step.

**Hypotheses:**

| | `exp2f` (now) | `ex2.approx.ftz`, predicted |
|---|---|---|
| element by element, every float x ≥ −126 | | **identical bit for bit** (same MUFU, the check never fires) |
| output on the test shapes | | **identical bit for bit** (no exponent there goes below −126) |
| loop SASS per tile | 734, 36 FSETP.GEU | **626** (−108), 0 FSETP.GEU, still 36 MUFU.EX2 |
| registers | 253, 0 spills | ≤ 253, 0 spills; still 2 blocks per SM |
| kernel time | 87.9 ms | **−3 … −7 %: 82–85 ms.** Not −15 %: issue slots are not the limit (34 % busy), the kernel waits on latency; the gain is the shorter softmax phase, in which the warp issues back to back |
| ncu | | issued instructions −15 %, tensor pipe a few points up, SFU work unchanged |

## M3.5 — result

| | `exp2f` (FAST_EXP2=0) | `ex2.approx.ftz` | predicted |
|---|---|---|---|
| all 2^32 floats, x ≥ −126 | | **0 differ** (1 703 935 differ, all in (−150, −126): subnormal vs 0; NaN equal) | held |
| output, 6 test shapes | | identical bit for bit | held |
| loop SASS per tile | 782 (counted from the loop label to the back branch) | **676 (−106)**: −36 FSETP, −72 predicated FMUL, +2 MOV; 36 MUFU.EX2 in both | held |
| registers, spills | 253, 0 | **255**, 0; still 2 blocks per SM | missed: one more, not fewer |
| kernel, 40 × 32760 (compare.py, 2 runs) | 80.6 / 86.3 ms | 79.5 / 85.2 ms: **−1.3 … −1.4 %** | **missed: −3 … −7 % predicted** |
| kernel, 4 × 32760 | 9.0 ms | 8.9 ms | |

Nsight Compute (4 heads, its fixed clocks):

| | `exp2f` | `ex2.approx.ftz` |
|---|---|---|
| duration | 7.79 ms | 7.71 ms (**−1.0 %**) |
| instructions executed | 3.083 G | 2.638 G (**−14.4 %**) |
| issue slots busy | 44.5 % | 38.4 % |
| tensor pipe / SFU (xu) / FMA / ALU | 62.0 / 17.4 / 16.9 / 16.4 % | 62.5 / 17.6 / 12.6 / 14.5 % |
| stalls per issued instruction: wait / math pipe / short scoreboard | 1.03 / 0.84 / 0.26 | 1.23 / 1.04 / 0.35 |
| warps per scheduler | 1.97 | 1.97 |

**Finding: 14 % fewer instructions buy 1 % of time.** The FSETP and FMULs were issued in slots in which the
warp would have waited anyway, for an exponent or a shuffle in flight. Take them out and the same wait
shows up as a stall instead (wait and math-pipe stalls per instruction go up by about as much as the
instruction count goes down). No pipe is near its limit, the SFU included. The time of a tile is set by the
chain of dependent latencies through it, with 2 warps per scheduler to cover it. That is direct evidence
for occupancy as the next lever, not instruction count. The change stays (default FAST_EXP2=1): it costs
nothing and frees issue slots that a third warp per scheduler can use. To be measured again after that.

## M3.6 — the bank conflicts of M3.4 (investigation)

**All conflict counters**, not only `op_ld` as in M3.4 (ncu, 4 × 32760):

| | `attention_int8pv` | `attention_int8pv_async` |
|---|---|---|
| conflicts, shared stores (`op_st`) | **11.17 M** | 0 |
| conflicts, shared loads (`op_ld`) | 0 | 218 K |
| conflicts, `op_ldgsts` | | 5.6 K |
| shared load wavefronts | 536 870 912 (= 2^29, 4 per `LDSM`, ideal) | 537 110 125 (+239 K, +0.045 %) |
| conflicts of type arbitration | 8.00 M | 0.29 M |

So M3.4 did not bring conflicts, it removed 98 % of them: the synchronous kernel's `STS.128` had 11 M,
which M3.4 missed by reading only `op_ld`. The rows the `LDGSTS` write in p_order_slot order are almost
clean (5.6 K): the suspect of M3.4 is cleared. The ldmatrix addresses are the same in both kernels, and the
synchronous one reads with no conflicts at all, so the 218 K are not a layout problem.

**Hypothesis:** they are arbitration. An `ldmatrix` and an `LDGSTS` write that lands at the same moment
compete for the same banks, and the read loses a cycle. The write can come from the same block (the next
tile lands while this one is read) or from the other block on the SM. Probe, with one block per SM (84
blocks: 84 × 64 queries, 32760 keys, 1 head) so that only the block's own writes remain:

| probe | predicted `op_ld` conflicts |
|---|---|
| `attention_int8pv_async`, 1 block per SM | a few tens of K: only collisions with its own next tile |
| the same, but waiting for the next tile right after issuing it (no overlap) | **≈ 0**, wavefronts = 4 per `LDSM` exactly |
| `attention_int8pv`, 1 block per SM | `op_st` conflicts **≈ 0**: its 11 M were stores colliding with the other block's ldmatrix |

Cost even if nothing is done: 239 K extra wavefronts over 84 SMs is ~2.8 K cycles per SM in a kernel of
~20 M cycles per SM, **≈ 0.01 % of the time**. Whatever the probe shows, a fix cannot pay back a change of
layout. The point of the probe is to know the cause.

**Result** (`bench/ncu_launch.py`, `sm__ctas_launched` confirms the blocks per SM):

| launch | `int8pv`: `op_st` / `op_ld` conflicts | `int8pv_async`: `op_ld` conflicts | load wavefronts per `LDSM` |
|---|---|---|---|
| 84 blocks, 1 per SM | 0 / 0 | **0** (`op_ldgsts` 41) | 4 exactly, both |
| 168 blocks, 2 per SM, started together | 0 / 0 | 11 K | 4 / 4.001 |
| 4 heads, 2560 blocks, ~25 per SM one after another | 11.17 M / 0 | 218 K | 4 / 4.002 |

- Alone on its SM, neither kernel has a single conflict. The block's own next tile, landing while it reads,
  costs nothing: the prediction of a few tens of K **missed**, and the probe that waits right after issuing
  is not needed.
- With two blocks that start together there are still almost none. They stay in step: both load, then both
  compute, so the stores of one never meet the reads of the other.
- The conflicts come when the two blocks on an SM are out of step, as they are once blocks finish and new
  ones start at different times. The stores (`STS`) or the landing copies of one block then meet the
  `ldmatrix` of the other. The prediction "arbitration, not layout" **held**, with this refinement.
- No fix: alone or in step the layout is conflict-free, and the cost is ≈ 0.01 %. Two things to keep in
  mind. With 3 blocks per SM (M3.7) there are more neighbours out of step, so this is to be measured again
  there. The phase between blocks may also be part of why the gain of M3.4 depended on the shape (not
  checked).

## M2 — instruction budget, the input for M3

**Instruction mix** (ncu, 4 heads): 537 M HMMA and 5.0 G warp instructions in all, **8.3 other
instructions per HMMA**; issue slots 24 % busy. Per warp and tile: 128 HMMA and ~1065 others, of which
**512 convert V from bf16 to fp16** (SASS of the kernel with and without the conversion: 1712 vs 1200).

**What it means for M3** (INT8 Q·Kᵀ, FP8 P·V, both m16n8k32): estimate, not yet measured.

- MMA: half as many instructions (k = 32), each at ~4× the rate (443 / 435 vs 111 TFLOPS): tensor floor
  ~50 ms, ~61 ms at today's 81.6 % efficiency.
- Other instructions per warp and tile: −512 (V quantized to FP8 once, before the kernel), + ~32 (S
  dequantization), + ~24 (P to FP8), + ~128 (flushing the FP16 P·V accumulator into FP32 every tile):
  ~1065 → ~740. Issue slots were 24 % busy at ~223 ms; with 64 MMA + ~740 others instead of
  128 + ~1065, issue alone takes ~37 ms.
- Exponentials stay: ~13 ms on the SFU.
- Tensor ~61 ms and issue ~37 ms overlap only partly with 8 warps per SM: **expect 65–85 ms**, about
  SageAttention 2 (68–76 ms) with less error. Beating it needs fewer non-MMA instructions, not only
  faster MMA.
