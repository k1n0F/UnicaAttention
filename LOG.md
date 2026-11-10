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

## M3.7 — more warps per SM, the cheap way (prediction, before measuring)

**What it attacks: latency, by giving each scheduler more warps to switch to.** M3.5 showed the kernel waits
on chains of dependent latencies with 2 warps per scheduler, and no pipe is near its limit (tensor 62 %,
issue 38 %). `MIN_BLOCKS_PER_SM=3` alone already brings ptxas to 168 registers with **no spills**, but three
blocks of 36.9 KB do not fit in the 100 KB of shared memory of an SM. With a 32-key tile the double buffers
take 18.4 KB, and only `-D` flags change.

**The price of a 32-key tile:** everything done once per tile is done twice per 64 keys. That is the flush
of the int32 accumulator into `out` (64 I2F, 64 FMUL, 64 FFMA), the max and its shuffles, alpha and beta,
the barrier. Loop SASS per 64 keys: 676 → 958 (+42 %). The P of each 32 keys gets its own max, so the
error changes a little; the emulator follows with `bc=32`.

| build | registers, spills | blocks (warps) per SM | loop SASS per 64 keys |
|---|---|---|---|
| K64 (now) | 255, 0 | 2 (8) | 676 |
| K64, MIN_BLOCKS 3 (control: the register cap alone) | 168, 0 | 2: shared memory | |
| K32 (control: the tile alone) | 231, 0 | 2 (8) | 958 |
| **A**: K32, MIN_BLOCKS 3 | 168, 0 | 3 (12) | 958 |
| **B**: K32, MIN_BLOCKS 4 | 128, 12 B outside the loop | 4 (16) | 1008 |

**Hypotheses** (bench/compare.py, kernel alone, all five in one run; ncu for the rest):

| | predicted |
|---|---|
| K64, MIN_BLOCKS 3 | **±3 %**: same warps, only the schedule ptxas can choose is tighter |
| K32 | **+10 … +25 %** slower: the per-tile overhead without anything to hide it |
| **A** | **−5 … −20 % vs K64** (center −12 %). If the time were set purely by latency, 3 warps instead of 2 would give 1.5× the throughput. The longer chain per 64 keys of K32 (~1.2×) and the tensor pipe ceiling (62 % now) eat part of it |
| **B** | **within 5 % of A**: a fourth warp helps less once the tensor pipe is near 75–80 %, and the code is longer |
| error | A and B: kernel vs emulator (bc = 32) < 0.1 %; rel L2 about the same 1.7 % |
| ncu, A | warps per scheduler 2 → 3, tensor pipe 62 → ~70–75 %, issue 38 → ~55 % |
| bank conflicts | more blocks out of step: more conflicts, still < 0.1 % of wavefronts |

## M3.7 — result

All five pass against the emulator (`bc` = their tile, kernel vs emulator ≤ 0.015 %). The 32-key tile is
even a little more accurate, since each 32 keys get their own P max: 1.66 % vs 1.69 % at 4 × 4096.

bench/compare.py, 40 × 32760, 5 rounds, kernel alone:

| build | kernel | vs K64 | predicted |
|---|---|---|---|
| K64 (now) | 88.2 ms | | |
| K64, MIN_BLOCKS 3 | 91.6 ms | +3.9 % | ±3 %: just outside |
| K32 | 103.1 ms | +16.9 % | +10 … +25 %: **held** |
| **A**: K32, 3 blocks | 92.1 ms | **+4.4 %** (−10.7 % vs K32) | −5 … −20 %: **missed** |
| **B**: K32, 4 blocks | 90.3 ms | **+2.4 %** (−12.4 % vs K32) | within 5 % of A: held |

Nsight Compute (4 heads; its durations wander between runs with the clock, so read them only within a run):

| | K64 | K32 | A | B |
|---|---|---|---|---|
| duration | 9.70 ms | 11.06 ms | 10.01 ms | 9.64 ms |
| warps per scheduler | 1.97 | 1.97 | **2.92** | **3.85** |
| instructions | 2.64 G | 3.82 G | 3.82 G | 3.97 G |
| issue slots busy | 38.4 % | 46.4 % | 55.7 % | 59.2 % |
| **tensor pipe** | **62.5 %** | 52.2 % | **62.7 %** | **63.9 %** |
| FMA / ALU / SFU | 14 / 15 / 18 % | 20 / 17 / 16 % | 24 / 21 / 20 % | 24 / 21 / 20 % |
| stalls per instruction: math pipe / wait / barrier / not selected | 1.04 / 1.23 / 0.17 / 0.40 | 0.41 / 1.11 / 0.27 / 0.41 | 0.89 / 0.92 / 0.47 / 0.79 | 1.35 / 1.22 / 0.51 / 1.07 |
| `op_ld` bank conflicts, of wavefronts | 214 K, 0.04 % | 118 K | 3.9 M, 0.7 % | 8.0 M, 1.5 % |

Calibration: on `peak_u8s8` (447 TOPS) ncu reads the tensor pipe at **99.6 %**, so 62–64 % is real idle time,
not a scale of the metric.

**Findings:**

- More warps do what they should, for the same tile: 2 → 3 → 4 warps per scheduler take K32 from 103.1 to
  92.1 and 90.3 ms, and its tensor pipe from 52 % to 63–64 %. But that only buys back the price of the
  smaller tile. Net, against K64: +4 % and +2 %. Neither A nor B is worth keeping.
- **The tensor pipe stops at ~63–64 %** whether a scheduler has 2, 3 or 4 warps, with issue slots, FMA, ALU
  and SFU all far from their limits. What grows with the warps is the math-pipe stall (0.41 → 0.89 → 1.35 in
  the K32 series): warps queue for the tensor pipe at some moments while it idles a third of the time at
  others. A guess, not checked: the warps of a scheduler bunch up in the same phase (all in MMA, then all
  in softmax) instead of interleaving.
- **The register cap alone costs ~4 %** (K64 with 168 registers, same 2 blocks).
- Bank conflicts grow with the blocks per SM as M3.6 led to expect, but faster than predicted: 1.5 % of
  wavefronts at 4 blocks (predicted < 0.1 %). Still small next to the rest.
- **For C** (K64, 3 blocks, swizzled tiles of ~33 KB): what these numbers predict is not encouraging.
  K64 is already at the same ~63 % tensor wall with 2 warps, and C would pay the ~4 % of the register
  cap. Expectation before any work: 0 … −5 %.

## M3.8 — the ~63 % tensor wall (investigation, prediction before measuring)

The question: why does the tensor pipe idle a third of the time whether a scheduler has 2, 3 or 4 warps?
Two explanations predict different things:

- **H1, the warps bunch up.** Each warp alternates an MMA phase (Q·Kᵀ: 32 IMMA, P·V: 32 IMMA) with a phase
  without MMA (max, exponents, packing, flush). If the warps of a scheduler drift into the same phase, they
  queue for the tensor pipe together and then leave it idle together; more warps only make the queue
  longer. Predicts: the math-pipe-throttle stalls fall on the IMMA instructions (≥ 70 % of them), and the
  time a warp spends waiting at an IMMA grows with the number of warps.
- **H2, one warp's chain.** Each warp has a fixed path per tile whose length is set by latencies (IMMA →
  max → shuffle → exponent → pack → IMMA), and the tensor pipe cannot run ahead of it. Predicts:
  the stalls are spread over the softmax instructions (wait, short scoreboard), not concentrated on IMMA.
  More warps should then have helped, which M3.7 contradicts. So H1 is the favourite.

**Measurement:** ncu source-level stall sampling (`SourceCounters`, `WarpStateStats`), samples per SASS
instruction, summed by phase: Q·Kᵀ, softmax, P·V with its flush, loads and loop control. K64 with 2 warps per
scheduler and B (K32, 4 blocks) with 4.

**Stall sampling, K64** (840 K samples): 96 % of the math-pipe stalls are on IMMA, as H1 predicts, but that
alone does not separate H1 from a plain tensor-bound burst. The SASS says more:

- The 32 Q·Kᵀ IMMA are issued back to back, one instruction apart, alternating accumulators, each with
  ~2600 "wait" samples: the fixed spacing between two IMMA of one warp, the pipe's own rate. No dependent
  chains. Within an MMA phase a warp already feeds the pipe as fast as it takes.
- ptxas reordered P·V: the 16 IMMA of step 0, then the 16 of step 1, then the flush, not pair by pair as the
  source reads.
- So the idle third of the tensor pipe lies entirely in the phases without MMA (softmax: ~200 instructions
  between the last Q·Kᵀ and the first P·V IMMA, flush, loads). The question becomes: why do the other
  warps of the scheduler not fill them?

**Timeline probe** (`UNICA_TRACE`): every warp on SM 0 records `clock64` per tile at four points, each tied by
a data dependency to the value that ends the phase: tile start, last Q·Kᵀ score converted, last P packed,
last `out` flushed. MMA phases are [start, scores] and [P packed, out flushed]. Per scheduler (`%warpid % 4`),
compare the time in which no warp is in an MMA phase with what independent warps would give, the product
of (1 − each warp's MMA fraction).

| | H1 (bunching) | independent phases |
|---|---|---|
| idle (no warp in MMA) vs the product | **clearly above**, ≥ 1.3× | ≈ 1× |
| two warps in MMA at once vs independent | above | ≈ 1× |

**Timeline result** (`bench/timeline.py`, 1 × 32760, SM 0):

| | K64 | K32 | A: K32, 3 blocks | B: K32, 4 blocks |
|---|---|---|---|---|
| resident warps per scheduler | 2.00 | 2.00 | 2.99 | 3.55 |
| MMA-phase fraction per warp | 0.63 | 0.54 | 0.61 | 0.83 |
| no warp in MMA: measured / if independent | **0.078 / 0.125** | **0.041 / 0.120** | 0.096 / 0.067 | 0.009 / 0.004 |
| 2+ warps in MMA: measured / if independent | 0.33 / 0.38 | 0.11 / 0.19 | 0.64 / 0.67 | 0.95 / 0.97 |
| median cycles per tile: Q·Kᵀ / softmax / P·V + flush / barrier + loads | 1002 / 751 / 1020 / 329 | 468 / 497 / 568 / 227 | | 682 / 649 / 1033 / 412 |
| whole tile, cycles | 3265 | 1724 | | 2905 |

**H1 is refuted.** With 2 warps the warps interleave better than chance: there is less time with no warp in
an MMA phase than independent warps would leave (0.078 vs 0.125). With 3 and 4 they are close to
independent. There is no bunching to break up.

What the timeline shows instead is one invariant: **~39 keys per 1000 cycles per scheduler** in every build:
2 × 64 / 3265, 2 × 32 / 1724 and 3.55 × 32 / 2905. The tensor pipe at its peak rate (16.2 cycles per IMMA per
scheduler, from `peak_u8s8`) would allow 62: 39 is the 63 %. A shared limit that does not move with warps
or tile size. A warp in its Q·Kᵀ phase takes 31 cycles per IMMA, twice the peak spacing.

**Next suspect: the MMA as the kernel issues it.** `peak_u8s8` gives every IMMA the same A register four
times and the same B register twice, with nothing in between. The kernel gives each IMMA four A
registers and a fresh B pair from ldmatrix, with ~10 other instructions per IMMA in the stream (676 / 64).
Microbenchmark `peak_u8s8_mixed_N` (csrc/peak.cu): distinct operand registers per chain, and N independent
FFMA per IMMA, at 2 and 4 warps per scheduler.

| | predicted, if the wall is the mix | if it is not |
|---|---|---|
| distinct registers, 0 fillers | ~peak | ~peak |
| 8–12 fillers per IMMA, 2 warps per scheduler | **~60–65 % of peak** | ≥ 90 % |

**Microbenchmark result** (`bench/peak.py`, one run per row group; runs drift ±10 % with the clock):

| per IMMA | 2 warps per scheduler | 4 warps per scheduler |
|---|---|---|
| nothing (`peak_u8s8`, one operand register) | 490 TOPS | 496 |
| distinct operand registers, nothing else | 493 | 498 |
| 4 FFMA | 460 (−7 %) | 465 |
| 8 FFMA | 402 (−19 %) | 409 |
| 12 FFMA | 342 (−31 %) | 350 |
| 16 FFMA | 294 (−40 %) | 300 |
| 8 FMUL | 402 | 405 |
| 8 FMNMX | 397 | 398 |
| 4 IADD3 (ptxas fused the 8 adds in pairs) | 357 | 378 |
| 8 I2FP | **488 (free)** | 495 |
| 8 MUFU.EX2 | 124 | 125 |

- Distinct registers cost nothing; the operand pattern of `peak_u8s8` is not what made it fast.
- **Other instructions take time away from the IMMA**, whatever the number of warps: 2 and 4 per
  scheduler give the same numbers. Up to about 4 per IMMA they almost fit in its shadow. Past that each
  FFMA, FMUL or FMNMX adds ~0.7–0.9 cycle to the 16 of the IMMA (8 → 16 fillers: 19.6 → 26.8 cycles per
  IMMA). IADD3 costs more per instruction. Prediction "~60–65 % at 8–12": held at 12 (69 %), 8 is higher
  (81 %).
- **Free:** I2FP (the int → float conversions of the flush and the scores). MUFU runs in its own pipe and
  limits only by its own rate (8 per IMMA = 64 cycles of SFU, so 25 %).
- A first version of the fillers (C++ `fmaf(x, 0.5f, 1.0f)`, chains from 0) gave 306 TOPS at 8 per IMMA
  instead of 402. Not explained and not reproduced. The cost per instruction above is uncertain by that
  much.
- This also explains M3.5. Its 108 removed instructions were 36 FSETP and 72 FMUL whose predicate was false
  on every lane. If the predicated ones cost the IMMA nothing, the 36 FSETP at ~0.5–1 cycle each come to
  0.6–1.1 %. 1.0 % was measured.

**Conclusion of M3.8.** The ~63 % wall is not bunched warps (refuted) and not too few warps (M3.7). It is
the work between the MMA, which shares the scheduler with them. Even perfectly mixed, the ~9.6 non-MMA
instructions per IMMA of the kernel would cost a good part of the pipe. The rest is the phase structure:
each warp has stretches of up to ~200 instructions with no IMMA ready (softmax), and the other warps then
cannot keep the pipe full at 4 + ~1 cycle per filler. The levers, in this light:

1. **Fewer FMA- and ALU-pipe instructions per IMMA**, above all in the flush (64 FMUL `out · alpha` + 64
   FFMA per tile) and the softmax. Free pipes (I2FP, MUFU) do not count.
2. **Overlap inside a warp** (Q·Kᵀ of tile t + 1 issued before the softmax of tile t). It shortens the
   stretches without IMMA, but adds no instructions only if the registers allow it.
3. **C** (K64 at 3 blocks) changes neither of the two. Expected ~0 % now, on firmer ground than in M3.7.

## M3.9 — lazy rescale of `out` (prediction, before implementation)

**What it attacks: FMA-pipe instructions per IMMA** (lever 1 of M3.8). The flush does
`out = fmaf(beta, I2F(acc), out · alpha)`, 64 FMUL per tile, whether alpha is 1 or not. alpha is exactly 1
when the running max of the row did not grow in this tile. On the benchmark's data (40 × 32760, random), a
warp sees a new max in some of its 16 rows in **12.1 % of the tiles** (62 of 512, counted in torch on 2 heads).
In the other 88 % the 64 FMUL multiply by 1.

**Change** (`UNICA_LAZY_RESCALE`, default 1; 0 is the M3.8 kernel): one warp vote per tile. If every lane
has alpha = 1 in both of its rows, `out` is left as it is. Otherwise `rescale_out` multiplies it first, the
same 64 FMUL, in a branch that the warp takes as a whole. The flush becomes `fmaf(beta, I2F(acc), out)`. Both
paths round exactly as before (`out · 1` is `out`), so the output does not change by a bit. The threshold
variant of FlashAttention 4 (keep a stale max until it is off by more than τ) would skip more, but changes
the rounding and the emulator. Not part of this step.

**Hypotheses:**

| | predicted |
|---|---|
| output | **identical bit for bit** to LAZY_RESCALE=0, on all test shapes |
| loop SASS | −64 FMUL on the path the warp usually takes, + ~4 (vote, compare, branch); the 64 FMUL move into a branch |
| instructions executed (ncu) | **−7 %** (0.88 · 64 − 4 ≈ 52 of 676 per tile) |
| registers | ~255, no spills |
| kernel time | **−1 … −2 %**: 52 FMA-pipe instructions per tile at ~0.5–1 cycle each, by M3.8's microbenchmark, of ~3265 cycles per tile. Small, but a direct test of the M3.8 model. compare.py drifts by ±2 %, so the cycle count of ncu decides |

## M3.9 — result

| | LAZY_RESCALE=0 | LAZY_RESCALE=1 | predicted |
|---|---|---|---|
| output | | identical bit for bit, 6 shapes × {random keys, keys growing along the sequence} × both u8 kernels. The rescale branch runs in 47 % / 96 % of their tiles | held |
| loop SASS | 676 | 673 static. The 64 FMUL sit in a branch (`VOTE.ALL`, `@P0 BRA` just before the P·V IMMA), so the usual path is **609** (−67) | held, −3 more than estimated |
| registers, spills | 255, 0 | **224**, 0; still 2 blocks per SM (3 need ≤ 168) | ~255: missed, in the good direction |
| instructions executed (ncu, 4 heads) | 2.638 G | 2.390 G (**−9.4 %**), FMA pipe 865 → 630 M (−27 %) | −7 %: held, a little more |
| tensor pipe | 62.5 % | 63.9 % | |
| active cycles per SM (ncu, 2 pairs of runs) | 20.45 / 20.54 M | 20.01 / 20.01 M: **−2.2 % / −2.6 %** | −1 … −2 %: **held, at the top and a little past it** |
| kernel, 40 × 32760 (compare.py, both orders) | 88.3 / 89.7 ms | 86.5 / 87.5 ms: **−2.0 % / −2.5 %** | |
| rel L2 | 1.73 % | 1.73 % | |

**Findings:**

- The model of M3.8 holds, and the removed FMUL were dearer than the microbenchmark said. ~59 instructions
  fewer per tile on average saved ~2.3 % of ~3265 cycles, **~1.2 cycles each**, against 0.5–0.9 in
  `peak_u8s8_mixed`. That matches the fit of M3.8 on the kernel itself (612 non-MMA instructions ≈ 614
  cycles), not the perfect mix of the microbenchmark. In the real kernel a non-MMA instruction costs about a
  cycle.
- **Under ncu the clock is not fixed from one process to the next**: 1.97–2.29 GHz in these four runs.
  Durations from separate ncu runs are not comparable. **Active cycles (`sm__cycles_active.avg`) are.**
  The ncu durations compared across processes in M3.4–M3.8 carry this uncertainty. Their ratios inside one
  report (pipe percentages, instruction counts) do not.
- The threshold variant (FlashAttention 4: rescale only when the max grows by more than τ) would remove
  the remaining 12 % of rescales and the vote. At ~1 cycle per instruction that is at most ~0.3 % more: not
  worth giving up bit-exactness with the emulator for.

## M3.10 — what the softmax and the loads do in every tile (prediction, before implementation)

The usual path of a tile (M3.9) is 609 instructions: 64 IMMA, 96 I2FP and 36 MUFU (free or on their own
pipes), and the rest. Reading the SASS, three things run in every tile but are needed in few or none:

| | per tile | why it is there |
|---|---|---|
| mask of the last tile: 32 `FSEL` + 16 `ISETP` | ~48 | `if (n_valid < KEYS_PER_TILE) mask_scores(...)`, but ptxas turned the branch into predicates: the mask is computed in every tile and selects nothing |
| bounds checks of the next tile's copies: `ISETP`, `SEL`, `IADD`, `MOV RZ` | ~15 | key < n_keys for each 16-byte copy, true for every tile but the last |
| moving the V fragments into pairs: `MOV` | ~48 | `ldmatrix` returns {r0, r1, r2, r3}, the IMMA wants {r0, r2} and {r1, r3} in adjacent registers |

**M3.10 (this step) takes the first two:**

- The last, partial tile goes through the loop body once more, after the loop, with the mask compiled in.
  Full tiles are compiled without it. Switch `UNICA_PEEL_LAST_TILE`, default 1.
- Copies past n_keys read the last key again instead of being zero-filled. Their scores are masked, so
  their P is 0 and the values do not matter (int8 has no NaN). One clamp per copy instead of a check and a
  select.

The pairs of V are the next step: they need a different V8 layout.

**Hypotheses:**

| | predicted |
|---|---|
| output | **identical bit for bit** (no value that reaches the output changes) |
| usual path SASS | 609 → **~550** (−48 mask, −~10 loads) |
| registers | ~224, no spills |
| kernel time | **−2 … −3 %**: ~58 instructions at ~1.2 cycles (M3.9) of ~3200 cycles per tile |

## M3.10 — result: the premise was wrong, nothing kept

- **The mask was never computed in full tiles.** It sits behind a real uniform branch
  (`BRA.U !UP0` around the `ISETP` / `FSEL` block), which the "usual path" count of M3.9 did not take out. My
  misreading of the SASS, not ptxas. The executed count says the same: ncu gives **569** instructions per
  warp and tile, not the 609 + 12 % · 64 = 617 of the static count. So the usual path of M3.9 is ~569, and
  the mask was never among them.
- What remained were the bounds checks of the copies, ~6 instructions per tile. Three builds, ncu (4
  heads, 2 runs each, identical within 0.1 %) and compare.py:

| build | active cycles | instructions | tensor pipe | compare.py kernel |
|---|---|---|---|---|
| M3.9 | 20.01 M | 2.390 G | 63.9 % | 86.4 / 87.1 ms |
| clamp, loop written as a lambda with the last tile peeled | 20.50 M (**+2.5 %**) | 2.349 G | 62.3 % | 88.0 / 88.1 ms |
| clamp, same lambda, last tile not peeled | 19.83 M (−0.8 %) | 2.361 G | 64.5 % | 84.6 ms (spread 8–14 %) |
| clamp, M3.9's loop as it was | 20.36 M (**+1.7 %**) | 2.365 G (−1.1 %) | 62.8 % | 88.0 / 88.1 ms |

- **Finding: ±2 % comes from how ptxas lays out the code, whatever the instruction count.** The same
  −1.1 % of instructions is −0.8 % in one build and +1.7 % in another, and the peeled loop, with the fewest
  instructions, is the slowest. A change of a few instructions per tile cannot be told from this
  scheduling noise. It is not kept: the code is back to M3.9. A step must remove clearly more than ~2 %
  of the instructions for its effect to be readable.
- Next by the same reading: the ~48 `MOV` that pair the V fragments for the P·V IMMA (~8 % of the
  executed instructions). Before predicting it, the executed count of what they cost will be checked
  first, not only the static SASS.

## M3.11 — V fragments in pairs (prediction, before implementation)

**Executed first** (ncu SourceCounters, per warp and tile): 569.8 instructions, of them **55 MOV**. The 48
before the P·V IMMA exist because `ldmatrix.m16n16.x2.trans.b8` returns {r0, r1, r2, r3} = (keys 0..15, dims
c and c + 8; keys 16..31, dims c and c + 8). The IMMA wants one B operand per block of 8 dims, keys 0..15
and 16..31 in two adjacent registers: {r0, r2} and {r1, r3}.

**What it attacks:** FMA/ALU-pipe instructions per IMMA, as M3.9 did.

**Change** (`UNICA_V_PAIRS`, default 1): the V tile in shared memory is laid out for the pairs. A row of a
32-key step holds, for each block of 8 dims, those 8 dims of key slot j and then of slot j + 16. The
"columns" c and c + 8 of the 16 × 16 matrix that ldmatrix transposes are then the same dims of slots j and
j + 16. r0, r1 is the B operand of one block of dims, r2, r3 that of the next. 16 rows of 256 bytes per
step, stride 272: rows shift by 4 banks, so 8 rows of one ldmatrix phase take 32 different banks. The V8
format in global memory does not change. Each 16-byte copy of a V row is split into two 8-byte copies
(`cp.async.ca`, 8 bytes; `.cg` takes only 16). +4 copies per thread and tile.
`quantize_v` and its tests stay as they are. A V8 layout that matches in global memory would save those 4
again, below the noise of M3.10.

**Hypotheses:**

| | predicted |
|---|---|
| output | **identical bit for bit** (same products, same order of the sums) |
| executed per warp-tile | 569.8 → **~526** (−48 MOV, +4 LDGSTS): −7.7 % |
| registers | ≤ 224, no spills |
| shared memory | unchanged (the new tile is 8704 bytes, inside the 9216 already reserved) |
| bank conflicts | `ldmatrix` none, as now; the 8-byte copies: some, from the two halves of a 16-byte chunk |
| kernel time | **−1.5 … −3 %**: 44 instructions at ~1.2 cycles is ~1.6 % of ~3265 cycles; more if the MOV were also on the path to the IMMA that read them |

## M3.11 — result: pairing at copy time loses, not kept

Two loaders tried, both bit-exact (the new test passed on all shapes, both u8 kernels, K64 and K32; the
exact single-tile P·V test passed too). ncu, 4 heads, 2 runs each, identical within 0.1 %:

| | executed instructions | active cycles | `LDGSTS` bank conflicts | tensor pipe | registers |
|---|---|---|---|---|---|
| V in rows of one key (M3.9) | 2.390 G | 20.01 M | ~1 K | 63.9 % | 224 |
| **1.** each 16-byte chunk of a key row split into two 8-byte copies | **2.193 G (−8.2 %)** | 20.52 M (**+2.5 %**) | **152 M** | 62.3 % | 242 |
| **2.** 8-byte pieces given out so that a warp writes one tile row of 256 bytes | 2.399 G (+0.4 %) | 20.42 M (+2.1 %) | 80 M | 62.6 % | 254 |

compare.py (40 × 32760, kernel alone, both orders) for loader 1: 84.9 / 85.2 ms against 84.9 / 85.1 ms.
No difference.

- **The MOV go away as predicted:** 48 fewer per tile, 569.8 → 523 executed (predicted ~526). The pairing
  itself is right.
- **The copies take it back.** Loader 1: the 8-byte writes of 8 threads land 32 bytes apart, two to a bank
  pair, and the 4 keys of a warp overlap on top. 152 M conflicts, ~36 extra wavefronts per tile, all on the
  shared-memory pipe the ldmatrix also use. Loader 2 halves the conflicts, but computes a key and two
  addresses for each of its 8 pieces in every tile: +48 instructions, as many as it saved. The addresses
  stay the same from tile to tile, but there are no registers left to keep them (254).
- **Prediction: missed.** −1.5 … −3 % predicted, +2 … +2.5 % measured: the cost of the 8-byte copies was
  not in the model.
- What would work: a V8 layout in **global memory** that already holds a tile row as it is wanted, so that
  16-byte copies stay as they are (4 per thread, no key arithmetic, no conflicts). That changes the V8
  format: `quantize_v`, V8 padded to whole tiles, its emulator test, the synchronous kernel and the
  single-tile test. Expected gain: the 48 MOV and nothing else, −1.5 … −3 %. Not started; the code is back
  to M3.9.

## M3.12 — Q·Kᵀ of the next tile under the softmax (prediction, before implementation)

**What it attacks: the stretches without MMA in each warp** (lever 2 of M3.8). Per tile a warp runs
Q·Kᵀ(t) (32 IMMA, ~1000 cycles in the timeline), then the softmax (~750 cycles, no IMMA), then P·V(t). The
softmax needs the scores of the same tile, so ptxas has nothing independent to put into the gaps between
the Q·Kᵀ IMMA, and nothing for the tensor pipe during the softmax. The other warp of the scheduler fills
part of that (tensor pipe 64 %), not all.

**Change** (new kernel `attention_int8pv_pipe`, the arithmetic of `attention_int8pv_async`): in iteration t
the warp issues Q·Kᵀ(t + 1), then the softmax of tile t, then P·V(t). Q·Kᵀ(t + 1) and the softmax of t are
independent, and ptxas interleaves them inside one basic block. Two details decide whether it can:

- **No branch between them.** The mask of a partial tile is applied to scores(t + 1) at the end of
  iteration t, not between the IMMA and the softmax. Q·Kᵀ(t + 1) is computed even in the last iteration,
  on stale shared memory, and thrown away.
- **Buffers:** iteration t reads K(t + 1) and V(t), and loads K(t + 2) into the K buffer of t (read by
  Q·Kᵀ(t) in the iteration before) and V(t + 1) into the other V buffer. Still one barrier per tile; the
  prologue loads K(0), V(0), K(1) and computes Q·Kᵀ(0).

**Hypotheses:**

| | predicted |
|---|---|
| output | **identical bit for bit** to `attention_int8pv_async` (same operations per tile, same order) |
| registers | a second set of 32 scores is alive across the softmax: 224 → **~250–255**. Spills are the main risk; if ptxas spills, the gain may vanish |
| executed instructions | about the same (one wasted Q·Kᵀ per warp out of 512) |
| SASS | the softmax instructions of tile t between the Q·Kᵀ IMMA of t + 1, not after them |
| tensor pipe | 64 % → **~70–75 %** |
| kernel time | **−5 … −15 %** (center −10 %). Up to ~750 cycles of softmax per tile could hide under the ~520 pipe cycles of Q·Kᵀ(t + 1) and between its IMMA, against ~3265 per tile. But the other warp already fills some of those gaps, and the softmax instructions placed between IMMA still cost ~1 cycle each past the first few (M3.8). |

## M3.12 — result (so far): the overlap works, the registers do not fit

The output is identical bit for bit to `attention_int8pv_async` on every shape, including 1, 2 and 3 tiles
and K32 (`test_pipelined_qk_changes_nothing_but_time`). In SASS the Q·Kᵀ IMMA of t + 1 now sit between the
max and exponent instructions of tile t, instead of in one block of 32.

Three versions. compare.py 40 × 32760 kernel alone, one run with all four; ncu 4 heads:

| | registers, spills | executed instructions | active cycles | tensor pipe | kernel |
|---|---|---|---|---|---|
| `attention_int8pv_async` K64 (M3.9) | 224, 0 | 2.390 G | 20.01 M | 63.8 % | 87.7 ms |
| pipe K64, plain loop with `scores = next` | 255, 0 | 2.710 G (+13.4 %) | 19.90 M (−0.6 %) | 64.3 % | 88.8 / 89.0 vs 88.2 / 88.5 ms (separate runs) |
| pipe K64, loop unrolled by 2 (no copy) | 255, **80 B spilled** | 2.501 G | 20.54 M (+2.6 %) | 62.3 % | 89.7 ms (+2.3 %) |
| async K32 | 196, 0 | 3.289 G | 21.94 M | 58.3 % | 94.6 ms |
| **pipe K32**, unrolled | 247, 0 | 3.127 G | **20.25 M (−7.7 % vs async K32)** | 63.1 % | **90.3 ms (−4.5 % vs async K32)** |

- **The mechanism holds.** Where registers suffice (K32), issuing Q·Kᵀ(t + 1) under the softmax saves 7.7 %
  of the cycles. In the plain K64 loop the stalls per instruction fell as predicted (wait 1.47 → 1.16, short
  scoreboard 0.74 → 0.36), but the copy `scores = next` (+42 MOV per tile) and buffer addresses computed per
  thread took it all back.
- **K64 runs out of registers.** Two sets of 32 scores, `out` (64), `q` (16), fragments and addresses do
  not fit in 255: the unrolled loop spills, and ptxas re-reads `threadIdx` (S2R) inside the loop instead of
  keeping it. Prediction "~250–255, spills the main risk": the risk came true.
- **Net: nothing beats `attention_int8pv_async` K64 yet.** Pipe K32 reaches it (+1 … +3 %): K32's own cost
  (+8 % against K64) eats what the overlap saves.
- What it would take for K64: ~20–30 registers fewer. Candidates: the Q fragments (16 registers) read from
  shared memory with `ldmatrix` each tile instead of being held (+4 LDSM per tile), or issuing only half of
  Q·Kᵀ(t + 1) under the softmax (16 extra scores instead of 32).

## M3.13 — Q from shared memory, to make room for the pipeline (prediction, before implementation)

**What it attacks: registers**, the limit M3.12 ran into. Each warp holds its 16 rows of int8 Q as 16
registers for the whole kernel. Instead it copies them once into shared memory (16 rows at the K8 stride,
2.3 KB per warp, 9.2 KB per block). Each Q·Kᵀ then reads them back with one `ldmatrix.x4` per 32-wide step
of d. `ldmatrix.x4` (b16, non-transposed) returns exactly the A fragment `load_q8` builds: row lane / 4,
bytes 4 · (lane % 4), matrices = rows 0–7 / 8–15 × bytes 0–15 / 16–31. Only one step's 4 registers are
alive at a time.

**Change** (`UNICA_Q_SMEM`, default 1, in `attention_int8pv_pipe` only): Q rows copied with `cp.async`
in the first group (rows past n_queries zero-filled), 4 `LDSM` per Q·Kᵀ.

**Hypotheses:**

| | predicted |
|---|---|
| output | identical bit for bit (the same bytes reach the same mma) |
| registers, pipe K64 | 255 with 80 B spilled → **≤ 255, no spills** (12–16 fewer needed) |
| shared memory | 36.9 → 46.1 KB per block; still 2 blocks per SM (2 × 47.1 KB ≤ 100 KB) |
| executed instructions | +4 LDSM per tile; the S2R / spill code of M3.12 gone |
| kernel time, pipe K64 vs async K64 | **−4 … −10 %**: the overlap saved 7.7 % of the cycles at K32, where it was not starved of registers. K64 has the same softmax to hide and fewer instructions around it. The +4 LDSM go to the memory pipe, not the FMA pipe |

## M3.13 — result

Bit for bit as before (`test_pipelined_qk_changes_nothing_but_time` and the emulator test pass with
Q_SMEM=1). All in one session; ncu 4 heads, compare.py 40 × 32760 kernel alone:

| | registers, spills | local loads (ncu) | executed | active cycles | tensor pipe | kernel (compare.py, 3 runs) |
|---|---|---|---|---|---|---|
| `attention_int8pv_async` K64 | 224, 0 | 0 | 2.390 G | 20.02 M (two runs) | 63.8 % | 85.2 / 84.6 / 85.2 ms |
| pipe K64, Q in registers (M3.12) | 255, 80 B | 33.6 M | 2.501 G | 20.53 M (+2.6 %) | 62.3 % | 87.0 ms |
| **pipe K64, Q from shared memory** | 255, **108 B** | 35.8 M | 2.506 G | **19.34 / 19.35 M (−3.4 %)** | **66.1 %** | **84.1 / 83.8 / 84.3 ms (−0.9 … −1.3 %)** |

- **The first build above the ~64 % wall:** the tensor pipe reaches 66.1 %, and the active cycles fall by
  3.4 %, repeatable to 0.1 %. The overlap of M3.12 now pays at K64.
- **But not for the reason predicted.** The spills did not go away: 108 B instead of 80 B, ~8.5 local
  loads per warp and tile in both builds. ptxas reads all four Q fragments in one block right before the
  Q·Kᵀ IMMA, so the peak of live registers did not drop. The spills sit at the top of the loop and after
  the rescale branch (values kept across the loop). What changed is where ptxas puts the work. With Q read
  inside Q·Kᵀ, it schedules the Q·Kᵀ of t + 1 with the softmax of t better, and the spill reloads cost
  less than they did. Prediction "no spills": **missed**. Time −4 … −10 %: **missed**, −3.4 % in cycles and
  −1 % in compare.py.
- **ncu and compare.py disagree** (−3.4 % vs −1 %). In M3.9 they agreed. The shapes differ (4 vs 40 heads)
  and so do the clocks (ncu's are not fixed, M3.9). Not explained yet. The lower number is the one to
  quote.
- The spills are the obvious next target: ~8.5 local loads per tile, and registers at 255.

**Where the gain comes from** (questions: register pressure, the place of the IMMA, shared
memory, and whether it is ptxas scheduling rather than the Q loads themselves).

nvdisasm (13.2 and 13.0) refuses life ranges for a kernel at 255 registers ("Invalid register count").
The live registers below are counted from the SASS of the loop instead: backward liveness with the back
edge, operand widths of IMMA / LDSM / 64-bit addresses, forward branches ignored. The count is below what
ptxas allocates: 209 live at most in `attention_int8pv_async` against its 224 registers, since it ignores
the alignment of register groups. The method is the same for all builds, so the comparison holds.

| | async K64 | pipe, Q in registers | pipe, Q from shared |
|---|---|---|---|
| live registers in the loop: max / mean | 209 / 161 | 220 / 190 | **217 / 180** |
| live across the back edge | 126 | 139 | **117** |
| live at the Q·Kᵀ IMMA | 162–179 | 180–197 | **169–185** |
| live at the P·V IMMA (the peak) | 150–209 | 196–217 | 183–217 |
| V `LDSM` → first use, instructions: mean / max | 21.9 / 47 | 20.4 / 40 | **39.9 / 159** |
| K/Q `LDSM` → first use: mean / max | 15.8 / 30 | 23.0 / 65 | 24.3 / 87 |
| Q·Kᵀ(t + 1): instructions spanned by its 32 IMMA; MUFU / FMNMX / FFMA inside | | 151; 24 / 25 / 25 | 161; 20 / 38 / 24 |
| spill reloads (LDL) per tile, static | 0 | 18 | 27 |
| stalls per instruction: wait / short scoreboard / MIO throttle / math pipe | 1.47 / 0.74 / 0.26 / 1.07 | 1.33 / 0.50 / 0.42 / 1.12 | **1.20 / 0.46 / 0.32 / 1.03** |
| shared memory per block; blocks per SM by shared memory / by registers | 36.9 KB; 2 / 2 | 36.9 KB; 2 / 2 | **46.1 KB; 2 / 2** |

1. **Register pressure fell, but not at the peak.** Q is no longer carried around the loop: 22 fewer
   registers live across the back edge, ~10 fewer in the mean, ~12 fewer around the Q·Kᵀ IMMA. The peak
   is in P·V (`out`, the P fragments, the V fragments and the scores of t + 1), and it did not move: 217
   against 220. That is why the spills stay.
2. **The Q·Kᵀ IMMA sit about the same.** They are interleaved with the softmax in both builds (151 and 161
   instructions for 32 IMMA, a similar mix inside).
3. **What moved are the V loads.** With the registers freed, ptxas issues the V `ldmatrix` about twice as
   far ahead of the P·V IMMA that use them: 40 instructions on average instead of 20, up to 159. Short-
   scoreboard and wait stalls fall with it. This matches the hypothesis: the gain is a different
   schedule that the lower pressure allows, not the Q loads. Those are 4 more LDSM per tile and cost.
   Not proven causally; the data point the same way.
4. **Shared memory:** 46.1 KB per block, 2 × 47.1 KB = 94.2 KB of the 100 KB per SM. **~2.9 KB per block
   of headroom.** One more buffer of K or V (9.2 KB) would drop the kernel to 1 block per SM. Registers
   also allow only 2 blocks, so both limits now bind together.

## M3.14 — what fills the P·V register peak (investigation)

The live registers at the peak of `attention_int8pv_pipe` (217, at a P·V IMMA), by the instruction that last
wrote each one (liveness from SASS as in M3.13):

| written by | registers | what |
|---|---|---|
| FMUL / FFMA | 65 | `out` |
| **MUFU** | **35** | **the 32 P of tile t as floats**, plus alpha and beta |
| I2FP | 32 | the scores of tile t + 1 |
| LDSM (V) | 24 | V fragments loaded ahead |
| IMMA (P·V) | 16 | accumulators |
| before the loop | 15 | pointers, scales |
| F2IP | 8 | P packed to u8 |

The 32 float P are already packed into 8 registers of u8 for the MMA. They stay alive for one reason: all
32 are next read by the `FADD` of the row sum, 7 to 698 instructions after the peak. The sum is needed
only by the next tile, so ptxas pushes it to the end to fill gaps. **32 registers at the peak for an
addition.**

Two ways out, measured before any kernel change:

- **A tree sum** (pairs of P right after their exponent, then 8 → 4 → 2 → 1). Same precision class, but
  a different order of float additions, so not bit for bit. Compiled and checked, not timed: spills grow
  from 108 / 204 to 148 / 268 bytes, the peak moves into Q·Kᵀ and stays at 255 allocated. **Worse,
  reverted.**
- **The row sum over the rounded P̂**, which the MMA uses anyway, with `IDP4A` (4 bytes · 1 per
  instruction, exact in int32): 8 instructions per tile instead of 32 `FADD`, and the float P die at the
  packing. It changes what the denominator is: Σ P̂ / 255 instead of Σ P. Emulator (`sum_rounded`, new
  option), rel L2 against FP32:

| data | Σ P (now) | Σ P̂ (rounded) |
|---|---|---|
| random, 4 × 4096 | 1.6968 % | 1.6968 % |
| random, 2 × 8192 | 1.7535 % | 1.7535 % |
| random, 1 × 16384 | 1.7414 % | 1.7415 % |
| sharper: q × 2, 4 × 4096 | 2.6486 % | **2.6999 %** |
| sharper: q × 4, 4 × 4096 | 3.2171 % | 3.2324 % |

  Neutral on the benchmark's random data, **+0.01 … +0.05 % absolute on sharper attention**. The many P
  that round to 0 still count in Σ P, and that turns out to be the better denominator. A choice of
  accuracy.

**Decision: the rounded sum, via `IDP4A`.**

## M3.15 — row sum over the rounded P̂ (prediction, before implementation)

**What it attacks:** the register peak of P·V in `attention_int8pv_pipe` (32 float P kept alive for the
row sum), and FMA-pipe instructions per IMMA (32 `FADD` per tile).

**Change** (`UNICA_SUM_ROUNDED`, default 1, all three u8 kernels): `online_softmax_u8` no longer sums the
float P. After packing, each lane sums its bytes with `__dp4a(p, 0x01010101, acc)`. Registers 0 and 2 of a
step hold row r, 1 and 3 row r + 8, so 4 `IDP4A` per row and tile, exact in int32. Then one I2F, and
`row_sum = row_sum · alpha + Σ P̂ · beta` as before. The scale is unchanged (P̂ is already 255 · P). The
emulator gets `sum_rounded=True` for these kernels.

**Hypotheses:**

| | predicted |
|---|---|
| error | rel L2 on the benchmark **1.73 %, unchanged** (emulator: ≤ 0.0001 % on random data); kernel vs emulator < 0.1 % |
| pipe K64: registers, spills | the 32 float P die at the packing: **no spills**, possibly < 255 registers |
| live at the P·V peak (SASS liveness) | 217 → **~185–190** |
| instructions per tile | −32 FADD + 8 IDP4A + 2 I2F: **−22**, ~−4 % |
| kernel time | pipe: **−2 … −5 %** (no spill reloads, fewer FMA-pipe instructions, and ptxas gets room again); async: −1 … −2 % (instructions only, it has no spills) |

## M3.15 — result

Tests pass: every u8 kernel against the emulator with `sum_rounded=True` (≤ 0.015 %), and both builds
against their own emulator on random and sharp (q · 4) attention (`test_both_denominators_match_their_emulator`).
compare.py 40 × 32760 (one run, all four), ncu 4 heads:

| | registers, spills | executed | local loads | active cycles | tensor pipe | kernel | rel L2 |
|---|---|---|---|---|---|---|---|
| async, Σ P | 224, 0 | 2.390 G | 0 | 20.01 M | 63.9 % | 85.2 ms | 1.73 % |
| async, **Σ P̂** | 224, 0 | 2.306 G (−3.5 %) | 0 | 19.57 M (**−2.2 %**) | 65.3 % | 83.6 ms (**−1.9 %**) | 1.73 % |
| pipe, Σ P | 255, 108 / 204 B | 2.506 G | 35.8 M | 19.35 M | 66.1 % | 84.3 ms | 1.73 % |
| pipe, **Σ P̂** | 255, **12 / 12 B** | 2.396 G (−4.4 %) | **24.6 K** | 18.87 M (**−2.5 %**) | **67.8 %** | **83.1 ms (−1.4 %)** | 1.73 % |

| | predicted | measured |
|---|---|---|
| error | 1.73 %, unchanged | **held** |
| pipe spills | none | 12 B left, local loads 35.8 M → 24.6 K: **held** |
| live at the P·V peak | ~185–190 | the peak moved to Q·Kᵀ and stays at **215**: **missed**. ptxas spent the freed registers at once: it loads 8 K/Q fragments ahead and keeps more from the prologue; still 255 allocated |
| instructions per tile | −22 | async −20, pipe −26: **held** |
| time, pipe | −2 … −5 % | −2.5 % cycles, −1.4 % compare.py: **held at the low end** |
| time, async | −1 … −2 % | −2.2 % cycles, −1.9 % compare.py: **held** |

**Where the u8 path stands now:** `attention_int8pv_pipe` at 83.1 ms kernel alone, 18.87 M cycles: −5.7 %
in cycles and −2.5 % in compare.py from `attention_int8pv_async` at the start of M3.12. Tensor pipe 67.8 %,
the highest so far.

## M3.16 — `attention_int8pv_pipe` becomes the default

Decision: `unica.attention()` runs the u8 path by default. The exact path stays available as
`kernel="attention_smem"` (0.17 %, the bf16 output floor). bench/harness.py, same run, 1 × 40 × 32760 × 128,
bf16 random inputs, median of 10, whole call (quantization included):

| kernel | time | speed-up | rel L2 | cos |
|---|---|---|---|---|
| PyTorch SDPA | 517.0 ms | 1.0× | 0.23 % | 0.999997 |
| SageAttention 2 | 74.3 ms | 7.0× | 3.90 % | 0.999238 |
| SageAttention 3 | 57.6 ms | 9.0× | 19.11 % | 0.981755 |
| **UnicaAttention** | **83.7 ms** | **6.2×** | **1.73 %** | **0.999850** |

Against the goal in the README: 2.25× less error than SageAttention 2, but 12.6 % slower than it. Not a
win yet. 74.3 ms means ~9.4 ms less, about 11 %. From M1 (~770 ms) that is 9.2× faster.

## M3.17 — TMA tile loads, V8 in P's order and in pairs (prediction, before implementation)

**What it attacks:** non-MMA instructions per tile, as M3.9 to M3.15 did, now in the loads. Executed per warp
and tile in `attention_int8pv_pipe` (ncu SourceCounters): 571.3, of them 8 `LDGSTS` with ~35–45
instructions of addresses and bounds checks around them, and 63.6 `MOV`, ~48 of which pair V fragments
(M3.11).

**Design (the full version):**

- **TMA** (`cp.async.bulk.tensor.2d`, SASS `UTMALDG.2D`; compiled and run on the card first): one thread
  issues a whole tile, an `mbarrier` with the expected byte count says when it has landed. The other 127
  threads compute no addresses at all.
- **No swizzle.** TMA writes its box densely, but the box may be wider than the tensor: the bytes outside
  come in as zeros. A box 144 bytes wide over rows of 128 gives the padded rows (K8_STRIDE) that all the
  ldmatrix addressing already uses, without a byte of padding in global memory. Measured with a probe
  kernel: rows padded with zeros, rows past the end zero, bit for bit. K8 stays as it is.
- **V8 changes format**, because TMA copies rows in order and P needs them in p_order_slot order.
  `V_PAIRS=0`: rows in P's order (`quantize_v` layout 1), box 144 × 64. `V_PAIRS=1`: rows in P's order and
  paired as in M3.11, each 256-byte row holding 8 dims of slot j, then of slot j + 16, for all 16 blocks of
  dims (layout 2). The box is 272 × 32, declared as uint16 (136 × 32) since a box dimension is at most 256
  elements. Both pad each head to whole tiles of 64 keys; keys past n_keys have P̂ = 0, so their bytes
  never count.
- New kernel `attention_int8pv_tma`, otherwise `attention_int8pv_pipe` (pipelined Q·Kᵀ, Q from shared
  memory, Σ P̂). Still one `__syncthreads` per tile, for the buffers about to be overwritten.

**Hypotheses** (each step measured on its own, same run):

| | predicted |
|---|---|
| output | **identical bit for bit** to `attention_int8pv_pipe`, both V8 formats |
| executed per warp-tile | 571 → **~530** with TMA, → **~485** with the pairs too (−15 %) |
| registers, spills | ≤ 255, no spills; fewer addresses alive |
| shared memory | ~46.1 KB + 3 mbarriers: still 2 blocks per SM |
| kernel time | TMA **−2 … −4 %**; the pairs another **−2 … −4 %**; together **−5 … −8 %** |

## M3.17 — result

**Bugs on the way.** The kernel first failed with `cudaErrorMisalignedAddress` in every TMA copy.
compute-sanitizer pointed at `tma_load_2d`, and a `printf` of the addresses showed the shared tiles aligned
(1024, 19456: multiples of 128) but the tensor map at `…388`: TMA needs it 64-byte aligned, and a map
declared right after the 8-byte Q8 pointer lands at offset 8 of the parameters. The probe kernel had the map
first. Fixed by putting both maps first. Also learned: `nvdisasm` life ranges and `compute-sanitizer` are
not on the default PATH (the latter is in the toolkit's `compute-sanitizer/`).

**Tests:** bit for bit equal to `attention_int8pv_pipe` on all shapes plus 1, 2 and 3 tiles, K64 and K32,
both V8 layouts (`test_tma_tiles_change_nothing_but_time`). Layouts 1 and 2 hold exactly the bytes of
layout 0 at the positions of `v8_offset` (`test_v_layouts_hold_the_same_bytes_in_p_order`). The emulator
test passes (≤ 0.015 %).

compare.py 40 × 32760 (one run, all three), ncu 4 heads:

| | registers, spills | executed per tile | `op_ld` bank conflicts | active cycles | tensor pipe | kernel |
|---|---|---|---|---|---|---|
| `attention_int8pv_pipe` | 255, 12 B | 571.3 | 2.77 M | 18.86 M | 67.8 % | 82.5 ms |
| **TMA**, V8 in P's order (V_PAIRS=0) | **235, 0** | 516.3 (−55) | 0.32 M | 17.36 M (**−8.0 %**) | 73.7 % | 76.0 ms (**−7.9 %**) |
| **TMA + pairs** (V_PAIRS=1) | 235, 0 | 467.3 (−49) | 0.28 M | 16.94 M (**−2.4 %**) | **75.5 %** | **74.0 ms (−2.6 %)** |

| | predicted | measured |
|---|---|---|
| output | bit for bit | **held** |
| executed per tile | ~530, then ~485 | 516, then 467: **held, a little better** |
| registers, spills | ≤ 255, none | 235, none: **held** |
| TMA | −2 … −4 % | **−8.0 %: missed, twice the gain** |
| pairs | −2 … −4 % | −2.4 … −2.6 %: **held** |
| together | −5 … −8 % | **−10.3 %** |

**Why TMA gave twice the prediction.** The prediction counted only the ~55 instructions of the copies. TMA
also took the copy addresses out of the registers: 255 → 235, and the last spills went. It also took the
`LDGSTS` writes out of the shared-memory pipe the ldmatrix use: bank conflicts 2.77 M → 0.32 M. Those
are the arbitration conflicts of M3.6, the copies of the other block meeting this block's reads. At ~1.2
cycles per instruction the copies alone explain ~2 %; the registers and the shared-memory pipe the rest.
Not separated further.

## M3.17 — `attention_int8pv_tma` is the default

Its output is identical bit for bit to `attention_int8pv_pipe`, the kernel made the default in
M3.16, so the accuracy decision stands. bench/harness.py, 1 × 40 × 32760 × 128, whole call, three runs:

| | run 1 | run 2 | run 3 | rel L2 |
|---|---|---|---|---|
| SageAttention 2 | 75.9 ms | 77.0 ms | 76.4 ms | 3.90 % |
| **UnicaAttention** | **76.4 ms** | **76.0 ms** | **76.6 ms** | **1.73 %** |

(SDPA 528.0 ms, SageAttention 3 58.1 ms at 19.11 % in run 1.)

**Level with SageAttention 2 in speed** (76.3 against 76.4 ms on average; the runs scatter by more than the
difference), at 2.25× less error. From M1 (~770 ms): 10.1×. The README goal, faster at no more error, is
not met yet: level is not faster.

## M3.18 — what `attention_int8pv_tma` waits on (investigation)

Executed per warp and tile (ncu SourceCounters, 4 heads): **467.0**. I2FP 98.1, FFMA 98.0, IMMA 64.1, FMNMX
38.0, LDSM 36.0, MUFU 36.0, F2IP 16.0, FMUL 13.1, IDP 8.0, the rest uniform-datapath and branch overhead.
Hardly a `MOV` left (1.9).

Stall samples (695 K) by instruction:

| instruction | share | top reasons |
|---|---|---|
| IMMA | 43.1 % | math pipe 57 %, wait 28 % |
| BRA + BSSY, i.e. the `__syncthreads` | **11.9 %** | barrier 71 % / 99 % |
| I2FP | 9.4 % | wait 42 %, selected 40 % |
| FMNMX | 7.1 % | short scoreboard 49 % |
| FFMA | 5.9 % | selected 62 % |
| LDSM | 4.0 % | selected 33 %, MIO 27 % |

By reason over everything: wait 26 %, math pipe 25 %, selected 18 %, **barrier 10 %**, short scoreboard 7 %.

- **The barrier is new as a target.** With TMA the per-tile `__syncthreads` only protects buffers about to be
  overwritten (WAR). Every warp of the block waits for the slowest one there, every tile.
- **No three-input max on sm_120.** `max.f32 d, a, b, c` is accepted by PTX, but ptxas emits two `FMNMX`
  (compiled and checked). The 32 FMNMX per tile stay.
- **Outside the kernel:** quantization is 2.9 ms of the 76.3 ms whole call (Q and K two passes each, a
  separate `k.mean`, a separate `aminmax` over V: ~2.2 GB at 810 GB/s ≈ 2.7 ms). Fusing the mean and the
  aminmax into the quantize kernels saves at most ~0.8 ms, ~1 %.

Order agreed: the barrier, then the quantization, then (with an accuracy check) the flush
every 128 keys.

## M3.19 — mbarriers instead of `__syncthreads` (prediction, before implementation)

**What it attacks:** warps of a block waiting for the slowest one every tile (barrier 10 % of the stall
samples). With TMA the `__syncthreads` only guards two buffers against being overwritten while another
warp still reads them.

**Change** (`UNICA_EMPTY_BARRIERS`, default 1, `attention_int8pv_tma`): four "empty" mbarriers, `k_free[2]`
and `v_free[2]`, each completed by 4 arrivals, one per warp (`__syncwarp`, then lane 0 arrives) after the
warp's last read of that buffer: after Q·Kᵀ(t + 1) for `k_free[later]`, after P·V(t) for `v_free[now]`.
Before issuing K(t + 2) and V(t + 1), thread 0 waits only for the arrivals that free those two buffers:
the reads of K(t) and of V(t − 1), phases t / 2 and (t − 1) / 2. No other warp waits for another.
Phases cannot run two ahead: the next phase of a free barrier needs the copy that thread 0 issues after its
wait. The re-arming of `landed` stays safe as well: every free arrival comes after that warp's wait on
`landed`.

**Hypotheses:**

| | predicted |
|---|---|
| output | **identical bit for bit** (the arithmetic does not change) |
| barrier stalls | 10 % of the samples → **~0**; warp 0 waits on the free barriers instead, a little |
| instructions per tile | about the same: −BAR.SYNC, +2 arrivals and the waits of thread 0 |
| registers | ~235, no spills |
| kernel time | **−2 … −5 %**: part of the barrier time is already covered by the other block's warps |

## M3.19 — result: slower, not kept

Bit for bit the same output (new test, 3 runs per shape, 1 to 4 tiles, K64 and K32), and `compute-sanitizer
--tool racecheck` reports 0 hazards (it may not see the TMA writes). Per-tile `BAR.SYNC` gone (5 → 2, the
prologue's), 7 `SYNCS.ARRIVE` added; registers 235 → 244, no spills.

| | `__syncthreads` | free mbarriers |
|---|---|---|
| kernel, compare.py (both orders) | 73.9 / 75.5 ms | 78.7 / 79.5 ms (**+6.5 %**) |
| active cycles (ncu, 2 runs each) | 16.93 / 16.94 M | 18.42 / 18.43 M (**+8.8 %**) |
| executed instructions | 1.960 G | 2.088 G (+6.5 %) |
| tensor pipe | 75.6 % | 69.4 % |
| stalls per instruction: barrier / long scoreboard / wait | 0.58 / 0.09 / 1.45 | **0.00 / 0.61** / 1.60 |

**Prediction missed, in the wrong direction.** The barrier stall went away and came back as a wait for data:
long scoreboard ×7 (the `try_wait` loop on `landed`), and ~30 more instructions per tile, the extra rounds of
that loop. The `__syncthreads` was not only a cost. It kept every warp in step with thread 0, which issues the
copies, so the next tile was always on its way in time. Without it, the fast warps run ahead and wait for
copies that thread 0 issues late: thread 0 is a computing warp itself, and it now also waits on the free
barriers for the slowest warp. FlashAttention 3 gives the copies a warp of their own. Here a fifth warp at
244 registers would leave 1 block per SM (2 × 160 × 244 > 64 K). Reverted.

## M3.20 — quantization: measured first

Each step alone (40 × 32760, median of 20):

| step | time | throughput |
|---|---|---|
| `quantize_int8(q)` | 0.640 ms | ~786 GB/s (read 335 MB + write 168 MB) |
| `k.mean` | 0.387 ms | 866 GB/s: at the limit |
| `quantize_int8(k, mean)` | 0.607 ms | as Q |
| `torch.aminmax(v)` | 0.398 ms | 843 GB/s: at the limit |
| **`quantize_v`, layout 2** | **0.998 ms** | **~504 GB/s** |
| all of `kernel_inputs` | 2.876 ms | |

The estimate of M3.18 was wrong about where the time goes. The separate passes for the K mean and the V
max run at the bandwidth limit, and they cannot go: a scale has to exist before anything is quantized with
it. What is slow is **`quantize_v` in the paired layout of M3.17**, a cost that M3.17 measured only the
kernel and missed. Each thread writes its 8 bytes with a stride of 16 bytes, so every 32-byte sector a warp
touches is left half written.

**Prediction:** `quantize_v_paired`, one thread per 16-byte cell of the output (dims 8b .. 8b + 7 of slot j
and of slot j + 16), so a warp writes 512 contiguous bytes and reads 256 contiguous bytes from each of two
keys. Padding keys become zeros (they were uninitialized). Output identical to layout 2
(`test_v_layouts_hold_the_same_bytes_in_p_order`). **0.998 → ~0.62 ms**, like the other quantize kernels;
whole call −0.4 ms, ~−0.5 %.

**Result: the premise was wrong, not kept.** The paired kernel passed the layout test, and changed nothing:

| `unica.quantize_v` | time |
|---|---|
| layout 0 (rows of one key, contiguous) | 0.992 ms |
| layout 1 | 0.988 ms |
| layout 2, quantize_v | 0.998 ms |
| layout 2, quantize_v_paired | 0.994 ms |

The Python `quantize_v` calls `torch.aminmax` itself. Its 0.998 ms already hold the 0.398 ms of the
aminmax, which the table above lists a second time. The quantize kernel of V takes ~0.55 ms, like the
others, and the write pattern of layout 2 costs nothing measurable. The parts of `kernel_inputs`:
0.64 (Q) + 0.39 (K mean) + 0.61 (K) + 0.40 (V max) + ~0.6 (V) ≈ 2.6 ms of 2.88, every pass near the
memory limit. What is left to save is one of the two reads of K and of V (~0.8 ms). The statistics pass and
the quantizing pass would have to run head group by head group, so that the second read hits L2 (a head
of K is 8.4 MB, L2 64 MB). That means ~40 launches instead of 5, or one cooperative kernel with grid syncs.

## M3.21 — the flush every 128 keys (checked before implementation, not started)

**Idea:** two tiles share one P max, so their P·V adds up in the same int32 accumulators (|sum| ≤ 128 · 255 ·
127 < 2^24, still exact), and `out` is flushed once per 128 keys: −32 I2F and −32 FFMA per tile.

**Accuracy** (emulator, `bc` = the keys sharing a P max, everything else as the kernel):

| data | bc = 64 (now) | bc = 128 | bc = 32 |
|---|---|---|---|
| random, 4 × 4096 / 2 × 8192 / 1 × 16384 | 1.714 / 1.754 / 1.766 % | 1.767 / 1.806 / 1.815 % (+0.05) | 1.677 / 1.717 / 1.732 % |
| sharper, q × 2 | 2.700 % | 2.860 % (+0.16) | 2.632 % |
| sharper, q × 4 | 3.232 % | 3.262 % (+0.03) | 3.221 % |

**Why it does not fit the kernel as it is:**

- The P of tile t would need the max of tile t + 1. `attention_int8pv_pipe` / `_tma` issue Q·Kᵀ(t + 1)
  precisely so that it runs under the softmax of t. With a shared max, the softmax of every even tile has
  to wait for Q·Kᵀ(t + 1), and the overlap is lost on half the tiles. Keeping the scores of three tiles
  instead does not fit in the registers (3 × 32 + `out` 64 + the rest > 255).
- Rough balance: the flush saves ~32 FMA-pipe instructions per tile (the I2FP are close to free, M3.8), at
  ~1.2 cycles each ≈ 2 % of a tile. Losing the overlap on half the tiles costs about half of what the overlap
  gave (M3.12–13: 3–8 % of the cycles). **Expected net ≈ 0 %, for +0.05 … +0.16 % error.** Not
  implemented.

## M3.22 — tile settings, again, on the TMA kernel (prediction, before measuring)

Every tile choice so far (K64, 4 warps, 2 blocks) was made on older kernels. `attention_int8pv_tma` has fewer
instructions per tile, no copy addresses, and different registers, so the balance may have moved. Compiled:

| build | registers, spills | shared memory | blocks (warps) per SM |
|---|---|---|---|
| K64, 4 warps (now) | 235, 0 | 46.1 KB | 2 (8) |
| K32, 4 warps | 188, 0 | 27.7 KB | 2 (8) |
| K32, 4 warps, MIN_BLOCKS 3 | 168, 36 / 72 B | 27.7 KB | **3 (12)** |
| K32, MIN_BLOCKS 4 | 128, 356 / 256 B | | heavy spills: skipped |
| 8 warps, K64 | 235, 0 | 55.3 KB | 1 (8) |
| 8 warps, K32 | 188, 0 | 36.9 KB | 1 (8) |
| 6 warps | | | does not compile (a tile does not split over 6 warps) |

**Hypotheses** (kernel alone, compare.py, all in one run):

| build | predicted vs K64 4 warps |
|---|---|
| K32, 4 warps | **+5 … +10 %**: same warps, twice the per-tile overhead (M3.7 saw +17 % before TMA took the copies out of it) |
| K32, 3 blocks | **−3 … +5 %**: 12 warps instead of 8 against the K32 overhead and a few spills; the most uncertain |
| 8 warps, K64 | **0 … +5 %**: same 8 warps, but one block, so the barrier holds 8 warps and no other block fills its gaps |
| 8 warps, K32 | **+5 … +15 %**: both costs |

Output: 8 warps K64 bit for bit equal to 4 warps (the same arithmetic per row); the K32 builds against the
emulator with `bc=32`.

## M3.22 — result: nothing to change, and the card is power-limited

All variants are correct (8 warps bit for bit; K32 within 0.1 % of the emulator at `bc=32`). compare.py,
kernel alone (one run with all five; its spread was 8–13 %, the clock was high that run):

| build | kernel | vs K64 4 warps | predicted |
|---|---|---|---|
| K64, 4 warps | 70.3 ms | | |
| K32, 4 warps | 83.4 ms | **+18.6 %** | +5 … +10 %: missed, worse |
| K32, 3 blocks | 78.8 ms | **+12.1 %** | −3 … +5 %: missed, worse |
| 8 warps, K64 | 67.1 ms | (noise; see below) | 0 … +5 % |
| 8 warps, K32 | 86.3 ms | +22.8 % | +5 … +15 %: worse |

K32 stays out: 12 warps per SM do not pay for twice the per-tile overhead, even with the copies gone.

**4 against 8 warps at K64, head to head:**

| | 4 warps | 8 warps |
|---|---|---|
| compare.py, both orders | 73.8 / 74.7 ms | 72.9 / 74.0 ms (−1 %) |
| ncu active cycles, 4 heads (2 runs) | 16.96 M | 18.05 / 18.07 M (**+6.5 %**) |
| ncu active cycles, 40 heads | 168.0 M | 180.3 M (**+7.3 %**) |
| tensor pipe (ncu) | 75.5 % | 70.9 % |
| bank conflicts `op_ld` | 0.27–0.29 M | ~20 (one block per SM: no other block to collide with, M3.6) |
| **SM clock / power, 6 s of calls (nvidia-smi)** | **2610 MHz / 360 W** | **~2800 MHz / 360 W** |
| time per call there | 73.4 / 74.8 ms | 73.7 / 73.6 ms |

**Finding: under full load the RTX 5080 runs at its power limit (360 W), not at a clock of its own.**
Both builds draw exactly 360 W. The 4-warp build needs 7 % fewer cycles but spends more energy per cycle (its
tensor pipe is busier), so the card clocks it 7 % lower: a tie in time. Consequences:

- ncu's cycle counts (taken at a clock ncu sets) predict wall time only while the power limit does not bind.
  At the benchmark shape it binds. Fewer cycles for the same work at the same energy buy nothing.
- What makes the kernel faster now is **less energy per attention**: fewer instructions, less data moved
  through registers and shared memory. M3.17 (−10 %) did that; packing the same work denser does not.
- Wall time under load (compare.py, harness.py) is the measure that counts. ncu stays the tool for *why*.

Nothing changes in the code: 4 warps K64 stays.

## M5.1 — a ComfyUI node for ComfyUI-WanVideoWrapper

The WAN 2.2 workflow (I2V A14B, high- and low-noise models, 832 × 480, 81 frames) runs on
ComfyUI-WanVideoWrapper, whose `WanVideoModelLoader` sets `attention_mode = "sageattn"`. WanVideoWrapper
already has an override path: `transformer_options["attention_mode_override"]` is passed by the model down
to the self-attention of each block, and from there to the `attention` function that `model.py` imports.

`comfyui/__init__.py`, node **Unica Attention (WanVideo)** (between a model loader and the sampler):

- clones the model and sets the override to `"unica"` for all steps and blocks;
- wraps that `attention` once: `"unica"` goes to `unica.attention`, from WanVideoWrapper's layout
  [batch, tokens, heads, 128] in any float dtype to bf16 [batch, heads, tokens, 128] and back (one copy each
  way); every other mode goes to the original;
- cross-attention to the text is not overridden by WanVideoWrapper, so it keeps the loader's mode;
- `measure_every` > 0: on every n-th self-attention call, unica and SageAttention 2 against FP32 on the first
  `measure_heads` heads of the real Q, K, V, appended to `build/wan_attention_errors.csv` (the real-data error
  measurement of PLAN.md, with nothing heavy written to disk).

Installed as a directory junction in `custom_nodes`. It runs in ComfyUI's own venv (torch 2.14.0+cu132), with
the cubins built beforehand in this repository: that shell has no nvcc. `tests/test_comfyui.py` checks the
wrapper without ComfyUI, against a stand-in module: "unica" equals `unica.attention` bit for bit in fp16
in/out, other modes pass through, and wrapping twice wraps once.

Known cost of this first version: the layout copies, ~3 × 335 MB in and 335 MB out per call, about 1.5 ms of a
~76 ms call. Reading WanVideoWrapper's layout directly in the quantize kernels and in store_out would remove
them.

## M5.2 — a node for ComfyUI's own WAN model

The Rigsy workflow runs WAN on ComfyUI's own nodes (`MODEL`), with `PathchSageAttentionKJ`. The
WanVideoWrapper workflow of M5.1 ran the system out of RAM: both 14B models offloaded to RAM with block
swap, and the text encoder on the CPU.

**Unica Attention (native)**, in the same `comfyui/__init__.py`, uses ComfyUI's
`transformer_options["optimized_attention_override"]`, which every attention call of ComfyUI's models goes
through (`wrap_attn` in `comfy/ldm/modules/attention.py`; for WAN's self-attention it also takes the tensors
out of their `AttentionTensorContainer` first). `PathchSageAttentionKJ` writes its own function into the same
key, so this node goes **after** it and keeps it as the previous override:

- self-attention (head_dim 128, as many keys as queries, no mask, default scale, `heads` passed by name) goes
  to unica, in both of ComfyUI's layouts ([b, s, heads · 128] and, with skip_reshape, [b, heads, s, 128]);
- every other call, including cross-attention to the text, goes unchanged to the previous override
  (SageAttention there) or to ComfyUI's own function;
- `measure_every` works as in M5.1, and the CSV now also has SageAttention 3 where it is installed.

`tests/test_comfyui.py`: self-attention equal to `unica.attention` bit for bit in both layouts; cross-attention,
a mask and a positional `heads` reach the previous override; patching twice installs it once. Measurement
smoke test in ComfyUI's venv, random 4 heads × 2048 tokens: unica 1.63 %, SageAttention 2 3.72 %,
SageAttention 3 18.7 %.

## M5.3 — first errors on real Q, K, V

Run by the Rigsy app's agent: the Rigsy WAN 2.2 I2V workflow (nvfp4 weights, lightx2v 4-step LoRA, normally
on SageAttention 3), Unica Attention (native) after the KJ patch on both models, seed 42, `measure_every`
20, `measure_heads` 2. 34 020 tokens per call. 160 self-attention calls = 4 steps × 40 blocks, so every 20th
call fell on the same two blocks: **block 19** (calls 20, 60, 100, 140) and **block 39**, the last (40, 80,
120, 160). rel L2 against FP32 on heads 0–1:

| block | unica | SageAttention 2 | SageAttention 3 | SageAttention 2 / unica |
|---|---|---|---|---|
| 19, 4 steps | **1.13 / 1.18 / 1.21 / 1.26 %** | 1.51 / 1.72 / 1.65 / 1.67 % | 10.1 / 11.1 / 10.9 / 11.2 % | **1.33 … 1.45×** |
| 39, 4 steps | 0.237 / 0.234 / 0.213 / 0.210 % | 0.226 / 0.241 / 0.211 / 0.203 % | 1.87 / 1.99 / 1.72 / 1.67 % | 0.95 … 1.03× |

cos: unica ≥ 0.99992, SageAttention 2 ≥ 0.99985, SageAttention 3 ≥ 0.9939.

- **Real attention is kinder than random inputs:** 0.2–1.3 % here against 1.73 % (unica) and 3.90 %
  (SageAttention 2) on random data. Random inputs remain the worst case, as PLAN.md intends.
- **The advantage over SageAttention 2 depends on the layer.** In block 19 unica is 1.33–1.45× more accurate.
  In the last block both sit at ~0.2 % and tie, close to what bf16 output rounding alone leaves (0.17 % on
  random data). SageAttention 3 is 8–9× worse than both throughout.
- **Coverage is thin:** 2 of 40 blocks, 2 of 40 heads. A step multiple of 40 lands on the same blocks every
  step. Next: `measure_every` coprime to 40 (e.g. 7 or 13), so the samples walk through all blocks, and more
  heads per call.
- The `*_ms` columns time 2 heads with a cold first call, and say nothing about speed.

## M5.4 — tile settings at the real shape, 34 020 tokens (prediction, before measuring)

The question: every tile choice was measured at 32 760 tokens, the real call has 34 020. What
changes:

| | 32 760 | 34 020 |
|---|---|---|
| K64 tiles per head | 512, the last with 56 keys | 532, the last with 36 keys |
| query blocks per head (64 rows) | 512 | 532 |
| blocks / slots (84 SMs × 2) | 20 480 / 168 = 121.9 waves | 21 280 / 168 = 126.7 waves |

**K128** does not compile: `K_SCALE_BLOCK` is 64 and a tile must share one K scale. It would need two K scales
per tile or K scales per 128 tokens (an accuracy change), and it would not fit anyway. Its tiles take ~83 KB
of shared memory, so 1 block (4 warps) per SM. Two sets of 64 scores in the pipeline plus `out` exceed 255
registers. Not measured.

**Hypotheses** (compare.py, 1 × 40 × 34 020, whole call and kernel alone, one run with all variants):

| build | predicted vs K64 4 warps |
|---|---|
| K64, 4 warps | the reference; 7.8 % more work than at 32 760 (34 020² / 32 760² = 1.078), so ~+7.8 % time |
| K32 | **+12 … +19 %**, as at 32 760 |
| K32, 3 blocks | **+10 … +13 %** |
| 8 warps, K64 | **a tie within ±2 %** (power limit, M3.22) |
| 8 warps, K32 | **+15 … +25 %** |

The partial last tile (36 of 64 keys) costs ~0.2 % of the tiles, the partly filled last wave of blocks ~0.3 %:
the order should not change.

**Result** (one run, all five, spread 0.8–2.3 %):

| build | whole call | kernel alone | vs K64 4 warps (whole / kernel) | predicted |
|---|---|---|---|---|
| **K64, 4 warps** | **81.2 ms** | **79.7 ms** | | |
| K32 | 101.1 ms | 98.5 ms | +24.5 % / +23.6 % | +12 … +19 %: missed, worse |
| K32, 3 blocks | 91.0 ms | 88.1 ms | +12.1 % / +10.5 % | +10 … +13 %: held |
| 8 warps, K64 | 81.5 ms | 78.8 ms | +0.4 % / −1.1 % | a tie: held |
| 8 warps, K32 | 102.2 ms | 98.5 ms | +25.9 % / +23.6 % | +15 … +25 %: held, at the edge |

- **The order is the same as at 32 760**, and K64 with 4 warps stays. 8 warps tie again, as the power limit
  predicts.
- The time scales with the work: 76.3 ms at 32 760 (M3.17, harness) × 1.078 = 82.3 ms, against 81.2 ms
  measured. Neither the partial tile nor the last wave shows.
- Quantization grows with the tokens, 2.9 → 3.0 ms, as a pass over memory should.

## M5.5 — speed over sequence length, and whole videos

**Over sequence length** (`bench/speed_chart.py`, 40 heads, whole call, harness timing, one session). A first
sweep ran while ComfyUI still held 6.9 GB of VRAM (models from the video runs): at 32 760 tokens
SageAttention 3 took 84 ms instead of 58, and UnicaAttention 87 instead of 76, the spill rigsy.dev/bench
warns about. After ComfyUI's `/free` (unload models), the sweep was run again:

| tokens | SDPA | SageAttention 2 | SageAttention 3 | UnicaAttention | Unica vs SA2 |
|---|---|---|---|---|---|
| 1 024 | 42 TOPS | 96 | 23 | 81 | −16 % |
| 2 048 | 43 | 182 | 86 | 179 | −2 % |
| 4 096 | 41 | 239 | 232 | 244 | +2.1 % |
| 8 192 | 41 | 244 | 325 | 256 | +4.9 % |
| 16 384 | 42 | 272 | 379 | 274 | +0.7 % |
| 32 760 | 42 (523.4 ms) | 290 (75.68 ms) | 381 (57.70 ms) | 292 (75.15 ms) | +0.7 % |

At 1–2K tokens the fixed cost of the separate quantization passes still weighs on so short a call.

**Whole videos** (the Rigsy workflow: WAN 2.2 I2V 14B NVFP4, lightx2v 4 steps, 81 frames at
480 × 864, seed 42; Unica Attention (native) nodes after `PathchSageAttentionKJ` on both models, enabled or
not; the KJ mode `sageattn_qk_int8_pv_fp8_cuda` for SageAttention 2, `sageattn3` otherwise). The variants
alternated: an identical prompt is served from ComfyUI's cache (one SageAttention 2 rerun took 1.2 s), and
the first run after the unload (68.7 s) included loading the models. Both are left out. Times from ComfyUI's
history:

| | runs |
|---|---|
| SageAttention 2 | 49.0 / 48.7 s |
| UnicaAttention | 49.5 / 48.9 / 48.8 s |
| SageAttention 3 | 45.7 / 45.2 s |

Earlier runs by the Rigsy app's agent: SageAttention 3 46.2 / 44.1 s, UnicaAttention 51.0 / 46.5 s. On a
4-step video UnicaAttention and SageAttention 2 tie; SageAttention 3 is ~3.5 s (7 %) faster. The three
videos diverge visibly on one seed (assets/compare.gif). The arithmetic differs, so the trajectory does too.
No ranking of quality follows from one seed.

## M4.1 — FP4 on sm_120a: which forms exist, and their rates

The plan: FP4 in one half of the tile at a time, Q·Kᵀ first, then P·V, then the other way round.
Before any kernel, what the hardware offers (compiled with nvcc 13.2, `-arch=sm_120a`):

| PTX | SASS |
|---|---|
| `m16n8k64 … kind::mxf4nvf4.block_scale.scale_vec::4X … ue4m3` (NVFP4: scale per 16) | `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` |
| `m16n8k64 … kind::mxf4nvf4` / `kind::mxf4 … scale_vec::2X … ue8m0` (MXFP4: scale per 32) | `OMMA.SF.16864.F32.E2M1.E2M1.E8` |
| the same with an f16 accumulator | **refused by ptxas** |
| `m16n8k32 … kind::f8f6f4 … e2m1.e4m3` (f32 or f16) | `QMMA.16832.F32/F16.E2M1.E4M3`: the FP8 unit, e2m1 kept in bytes |
| `m16n8k32 … kind::mxf8f6f4.block_scale … ue8m0` | `QMMA.SF.16832.F32.*.E8` |

**Hypotheses:** OMMA ~900 TFLOPS, 2× INT8; if its f32 accumulator halves the rate as FP8's does, ~450 and FP4
buys nothing. QMMA with e2m1 as FP8: ~226 (f32) / ~456 (f16). Half the work in FP4, half in 8-bit: the
harmonic mean, ~600.

**Measured** (bench/peak.py, two runs; the clock was high in the first):

| form | run 1 | run 2 | predicted |
|---|---|---|---|
| `peak_u8s8` (today's mma, reference) | 496 TOPS | 446 | |
| `peak_e4m3_f32` / `_f16` | 246 / 496 | 223 / 454 | |
| **`peak_nvf4`** | **990 TFLOPS** | **931** | ~900: **held**, 2.0–2.1× INT8 |
| `peak_mxf4` | 991 | | the same: held |
| `peak_e2m1_e4m3_f32` / `_f16` | 245 / 495 | | held: FP8 rates |
| **`peak_mxf8f6f4_e2m1_e4m3`** (block-scaled, f32) | **494** | 452 | ~226: **missed**, full rate |
| **`peak_mxf8f6f4_e4m3_e4m3`** (block-scaled, f32) | | **446** | full rate |
| `peak_pair_nvf4_s8s8` / `_u8s8` (half the work each way) | 643 / 656 | | ~600: held (harmonic mean of run 1: 661) |

- **Only the block-scaled k64 form doubles the rate**, and it needs FP4 on both sides of the mma. FP4 against
  FP8 goes through the FP8 unit at FP8's rate.
- **Not predicted:** a block-scaled QMMA accumulates in f32 at full rate, where the plain one runs at half. So
  e4m3 P·V with an f32 accumulator no longer costs half the rate. M3.3 ruled e4m3 P·V out on accuracy, not on
  this, but the f32 accumulator would remove the flush. Kept for later.

## M4.2 — FP4 accuracy, emulator (before any kernel)

`bench/emulate.py` got FP4: `to_e2m1` (nearest, ties to the even mantissa, saturating at 6) and
`fp4_blocks` (one scale per 16 or 32 elements along the summed axis: d for Q and K, keys for P and V).
NVFP4: amax / 6 rounded to e4m3 under one FP32 scale per head. MXFP4: the OCP rule, 2^(floor(log2 amax) − 2).
Each config is today's kernel (INT8 Q·Kᵀ, u8 × s8 P·V, Σ P̂) with one half replaced.

**Hypotheses:** the output error is ~0.92 × the score error (the model that gives INT8 Q·Kᵀ's 1.33 %), and
e2m1 with a scale per 16 has a relative RMS error of ~9 %. Q·Kᵀ in NVFP4 **~12 %**, MXFP4 ~15 %; P·V in NVFP4
**~12–13 %**; both in FP4 **~17–19 %** (SageAttention 3 measures 19.1 %). My intuition: Q·Kᵀ is the better
half to start with, since P had been the most sensitive operand (M3.3).

**Measured**, rel L2 against FP32:

| config | 4 × 4096 | 2 × 8192 | q × 2 | q × 4 |
|---|---|---|---|---|
| today: INT8 Q·Kᵀ + u8 × s8 P·V | 1.70 % | 1.77 % | 2.65 % | 3.17 % |
| Q·Kᵀ NVFP4 alone | 14.00 % | 14.26 % | 25.80 % | 32.77 % |
| Q·Kᵀ MXFP4 alone | 16.93 % | 17.29 % | 31.13 % | 38.83 % |
| P NVFP4 alone (tile-local max) | 9.25 % | 9.43 % | 3.67 % | 0.83 % |
| V NVFP4 alone | 9.81 % | 9.86 % | 9.50 % | 9.50 % |
| **a:** Q·Kᵀ NVFP4 + u8 × s8 P·V | 14.04 % | 14.31 % | 25.90 % | 32.82 % |
| a: Q·Kᵀ MXFP4 + u8 × s8 P·V | 16.96 % | 17.33 % | 31.14 % | 38.87 % |
| **b:** INT8 Q·Kᵀ + NVFP4 P·V | **13.67 %** | **13.81 %** | **11.38 %** | **10.04 %** |
| b: INT8 Q·Kᵀ + MXFP4 P·V | 19.15 % | 19.36 % | 15.90 % | 12.02 % |
| c: NVFP4 both | 19.53 % | 19.86 % | 28.52 % | 34.14 % |

| | predicted | measured |
|---|---|---|
| Q·Kᵀ NVFP4 | ~12 % | 14.0 %: held, a little worse |
| Q·Kᵀ MXFP4 | ~15 % | 16.9 %: held, a little worse |
| P·V NVFP4 | ~12–13 % | 13.7 %: held |
| both NVFP4 | ~17–19 % | 19.5 %: held; the emulator reproduces SageAttention 3's 19.1 % |
| start with Q·Kᵀ | | **wrong** |

**Findings:**

- **Q·Kᵀ is the worst half for FP4**, not P·V. The error of a score grows with its size, and exp amplifies it:
  14 % on random data, 26 % and 33 % on sharper attention.
- **In P·V the error comes from V, not from P.** P alone costs 9 % on random data and less the sharper the
  attention (0.8 % at q × 4): the large P are few and land on the grid near 1. V costs 9.5–9.9 % everywhere.
  The mma takes e2m1 on both sides, so P cannot go to FP4 alone.
- **b beats a on every data set**, and beats both-FP4 (SageAttention 3's class) by 1.4–3.4×. Neither comes
  near SageAttention 2's 3.9 %.
- **NVFP4 beats MXFP4 everywhere** (a scale per 16 in e4m3 against a power of two per 32).

## M4.3 — correcting V's FP4 residual (checked in the emulator, not built)

**Idea:** V = V₁ + R, V₁ in NVFP4, so P·V = P·V₁ + P·R. Correct with all of R, or
only with its largest part. Random 4 × 4096, P in NVFP4 in every row of the table:

| | q × 1 | q × 2 | q × 4 |
|---|---|---|---|
| today (u8 × s8) | 1.70 % | 2.65 % | 3.17 % |
| b: V in NVFP4 | 13.67 % | 11.38 % | 10.04 % |
| V exact (what P alone costs) | 9.48 % | | |
| V₁ + the largest 1 / 2 / 4 of each 16 residuals, exact | 12.30 / 11.47 / 10.62 % | | |
| **V₁ + R in NVFP4 too (a second OMMA)** | 9.52 % | 6.07 % | 3.33 % |

V alone: 9.50 % → 0.87 % with the second FP4 term.

- **The full residual buys back exactly what FP4 saved.** Two OMMA at 2× the rate take the tensor time of one
  8-bit mma, and P stays in FP4. Today's speed, and worse than today on every data set.
- **The residual is not sparse.** The largest 1 of 16 holds 37 % of its energy, 2 hold 56 %, 4 hold 76 %:
  rounding noise is spread over every element. A partial correction cannot run on the tensor cores (an mma
  is dense), and on the FMA pipe the largest 1 of 16 alone is ~256 FFMA per lane and tile, against 467
  instructions in today's whole tile (M3.8: ~1.2 cycles each). For 13.7 → 12.3 %.
- Not pursued. The cheap kind of correction is a compact part that is exact: the per-channel mean of V
  (o = P·(V − μ) + μ, since a row of P sums to 1). Zero on random data by construction; to be seen on real V.

## M4.4 — centred V, FP4 P × FP8 V, two FP4 terms (emulator)

Three ideas. The price in tensor time of P·V, against today's u8 × s8 (1.0×): one OMMA k64 takes
the time of one IMMA k32 and covers twice the keys, so each FP4 term costs 0.5×. e2m1 × e4m3 runs as QMMA k32:
1.0×. Only `kind::mxf8f6f4` takes e2m1 × e4m3 with block scales, and its scales are ue8m0 per 32 (MXFP4 P,
MXFP8 V). New in the emulator: `smooth_v`, `p_terms` / `v_terms` (x ≈ x₀ + x₁, x₁ the same format of the
residual), `p1_v1`, the format `mxfp8`. Before the run only the price table was written down, no error
numbers, so these rows are an exploration, not tested hypotheses.

4 × 4096, rel L2. The second block adds to V a constant per channel (2σ, random per channel): the case real V
with channel offsets would be. There the output is dominated by that constant, so its errors are small; compare
them only within the block.

| | P·V tensor | random q × 1 / × 2 / × 4 | V with channel offsets q × 1 / × 2 / × 4 |
|---|---|---|---|
| today, u8 × s8 | 1.0× | 1.70 / 2.65 / 3.17 % | 0.16 / 0.22 / 0.75 % |
| b: NVFP4 P × NVFP4 V | 0.5× | 13.67 / 11.38 / 10.04 % | 0.96 / 1.53 / 5.15 % |
| **b + centred V** | **0.5×** | 13.54 / 11.38 / 10.05 % | **0.24 / 0.61 / 2.25 %** |
| MXFP4 P × MXFP8 V | 1.0× | 15.70 / 11.90 / 4.77 % | 0.33 / 0.73 / 1.65 % |
| MXFP8 P × MXFP8 V | 1.0× | 4.07 / 3.89 / 4.23 % | 0.27 / 0.43 / 1.57 % |
| P 1 term × V 2 terms | 1.0× | 9.52 / 6.07 / 3.33 % | 0.20 / 0.37 / 0.88 % |
| P 2 terms × V 1 term | 1.0× | 9.94 / 9.80 / 9.96 % | 0.95 / 1.47 / 5.12 % |
| P 2 × V 2, without P₁V₁ | 1.5× | 2.02 / 2.57 / 3.15 % | 0.17 / 0.24 / 0.84 % |
| P 2 × V 2, all four | 2.0× | 1.82 / 2.55 / 3.15 % | 0.17 / 0.23 / 0.84 % |

- **Centring V is free and exact, and it helps exactly when V has channel offsets**: b's error falls 4× there
  (0.96 → 0.24 %), and not at all on random data. Whether real V has them decides its worth.
- **FP4 P × FP8 V loses on both counts**: today's tensor time, and worse than b on random data, since the only
  scaled form forces MXFP4 on P.
- **Two FP4 terms never beat u8 × s8 at the same tensor time.** At 1.0× the best is 9.5 % against 1.7 %; today's
  accuracy needs 1.5×. The hardware sells precision by the bit: an FP4 term is half the price of an 8-bit one and
  holds half the bits, and on P, which lies in [0, 1], the uniform grid of u8 spends its bits better than two
  e2m1 terms with their scales.
- So FP4 pays only where fewer bits suffice: a layer that tolerates ~10 %, or data made friendlier first
  (centring). Both are questions for real Q, K, V.

## M4.5 — rotating Q and K (Hadamard) before quantization (emulator)

The idea: a transform of Q and K that keeps every q·k. An orthogonal H does: (q·H)·(k·H) = q·H·Hᵀ·kᵀ = q·k.
The Hadamard matrix (QuaRot, SpinQuant) mixes the channels of d, so an outlier is spread over all of them.
`rotate_qk` in the emulator: H block-diagonal in blocks of 16 or 128 channels, applied after the K mean.

**Hypotheses:** random data **~0** (a rotated Gaussian is Gaussian). With channels of uneven size (WAN has RMSNorm
with learned weights on q and k) a clear gain **for FP4**, and also **for INT8**, whose one scale per token covers all
128 channels, so an outlier sets it for all.

Synthetic channels: each channel of Q and of K times a factor (lognormal σ = 0.5 or 1.0, or 4 channels × 10),
normalised so that the scores keep the size of plain random data. 4 × 4096, rel L2, q × 1 / × 2 / × 4:

| channels | INT8 Q·Kᵀ (today) | + Hadamard 128 | Q·Kᵀ NVFP4 | + Hadamard 16 | + Hadamard 128 |
|---|---|---|---|---|---|
| random | 1.70 / 2.65 / 3.17 % | 1.70 / 2.68 / 3.23 % | 14.0 / 25.9 / 32.8 % | 14.0 / 25.7 / 32.7 % | 14.0 / 26.0 / 32.6 % |
| lognormal 0.5 | 2.87 / 4.83 / 6.12 % | **1.75 / 2.71 / 3.28 %** | 14.3 / 26.5 / 32.8 % | 14.7 / 27.0 / 33.7 % | 14.9 / 27.3 / 34.5 % |
| lognormal 1.0 | 5.69 / 8.46 / 12.27 % | **1.93 / 2.71 / 3.79 %** | 17.4 / 27.5 / 36.1 % | 18.8 / 29.6 / 38.3 % | 18.8 / 30.1 / 41.5 % |
| 4 channels × 10 | 8.41 / 14.69 / 18.47 % | **2.34 / 3.79 / 4.65 %** | 16.9 / 32.5 / 39.0 % | 17.2 / 33.2 / 39.5 % | 27.4 / 51.9 / 58.7 % |

| | predicted | measured |
|---|---|---|
| random | ~0 | ~0: **held** |
| INT8, uneven channels | a clear gain | **held, strongly**: 3–4× less error, back near the random-data level |
| FP4, uneven channels | a clear gain | **wrong: worse**, up to 1.6× with 4 outliers and H128 |

**Why it helps INT8 and hurts FP4.** INT8 has one scale for all 128 channels of a token: an outlier sets it, and
every other channel is left with a few levels. FP4 has a scale per 16 channels: an outlier is the maximum of its
own block and lands on the grid's top value nearly exactly, and the blocks without it keep their own scale. Fine
block scales already handle channel outliers. The rotation then mixes the outlier into every element, each with
e2m1's ~9 % error, and the term that dominates q·k loses the precision it had for free. A rotation makes data
Gaussian, and Gaussian is the worst case for FP4.

**For today's kernel:** if real Q and K have uneven channels, a Hadamard rotation in the quantize pass could cut
the INT8 Q·Kᵀ error several times, with the attention kernel unchanged (the scores are the same). The quantize pass
is memory-bound (M3.20), and a 128-point fast Hadamard transform is 7 butterfly steps per token. Real data decides
here too: on block 19 unica measured 1.13–1.26 %, below random's 1.70 %, which does not suggest strong outliers
there.

## M4.6 — the candidates on real Q, K, V: 22 blocks of a WAN 2.2 video

Run by the ComfyUI app's agent: the Rigsy WAN 2.2 I2V workflow as in M5.3 and M5.5 (nvfp4 high and low noise
models, lightx2v 4 steps, 480 × 864, 81 frames, 34 020 tokens, seed 42; Unica Attention (native) after
`PathchSageAttentionKJ` in sageattn3 mode). `measure_every` 7, coprime to 40, so the 22 samples fall on 22
different blocks over all 4 steps; 2 heads each. The node now also emulates the candidates of M4.4–M4.5 on the
same Q, K, V and records how uneven the channels are (largest channel RMS / median, 1 = even) and V's offset
(RMS of the channel means / RMS of V). `build/wan_attention_errors_m4.csv`. rel L2 against FP32:

| step, block | unica | + Hadamard | SA2 | SA3 | FP4 P·V | + centred V | channel spread q / k | V offset |
|---|---|---|---|---|---|---|---|---|
| 0, 6 | 0.662 | 0.605 | 0.708 | 5.50 | 2.38 | 2.29 | 1.73 / 2.39 | 0.341 |
| 0, 13 | 1.111 | 0.984 | 1.062 | 8.70 | 3.62 | 3.43 | 2.40 / 1.92 | 0.257 |
| 0, 20 | 1.151 | 1.076 | 1.569 | 9.36 | 5.53 | 5.47 | 1.79 / 1.79 | 0.183 |
| 0, 27 | 0.843 | 0.814 | 0.873 | 7.46 | 3.66 | 3.38 | 1.62 / 1.72 | 0.226 |
| 0, 34 | 0.764 | 0.661 | 1.001 | 7.12 | 3.48 | 3.37 | 2.65 / 4.20 | 0.330 |
| 1, 1 | 1.108 | 0.977 | 2.326 | 11.95 | 8.30 | 8.07 | **5.96 / 10.74** | 0.226 |
| 1, 8 | 0.920 | 0.834 | 1.584 | 8.63 | 5.01 | 4.79 | 2.15 / 2.32 | 0.282 |
| 1, 15 | 1.076 | 0.900 | 1.501 | 9.19 | 4.45 | 4.39 | 2.79 / 2.25 | 0.247 |
| 1, 22 | 1.231 | 1.088 | 1.736 | 10.36 | 5.26 | 5.15 | 1.97 / 3.20 | 0.174 |
| 1, 29 | 0.583 | 0.555 | 0.781 | 5.70 | 2.58 | 2.45 | 2.30 / 2.03 | 0.314 |
| 1, 36 | 0.352 | 0.303 | 0.437 | 3.46 | 1.47 | 1.37 | 1.95 / 3.33 | 0.268 |
| 2, 3 | 1.277 | 1.051 | 2.361 | 12.65 | 8.11 | 7.91 | **4.48 / 6.78** | 0.222 |
| 2, 10 | 0.658 | 0.610 | 0.836 | 5.76 | 2.64 | 2.61 | 1.82 / 1.58 | 0.266 |
| 2, 17 | 1.361 | 1.049 | 1.662 | 11.44 | 4.22 | 4.13 | 2.58 / 2.65 | 0.206 |
| 2, 24 | 0.760 | 0.722 | 0.943 | 6.69 | 3.20 | 3.12 | 1.98 / 1.82 | 0.253 |
| 2, 31 | 1.012 | 0.907 | 1.251 | 9.45 | 3.67 | 3.52 | 2.37 / 2.66 | 0.277 |
| 2, 38 | 0.176 | 0.175 | 0.180 | 2.08 | 0.81 | 0.55 | 1.53 / 1.50 | 0.327 |
| 3, 5 | 0.684 | 0.647 | 0.851 | 6.35 | 2.88 | 2.85 | 2.18 / 2.57 | 0.225 |
| 3, 12 | 0.629 | 0.596 | 0.595 | 4.95 | 2.48 | 2.44 | 1.92 / 1.63 | 0.237 |
| 3, 19 | 1.263 | 1.107 | 1.674 | 11.18 | 4.75 | 4.65 | 2.11 / 2.06 | 0.228 |
| 3, 26 | 1.353 | 1.161 | 2.068 | 11.48 | 6.32 | 6.18 | 1.88 / 1.81 | 0.231 |
| 3, 33 | 0.876 | 0.754 | 1.830 | 14.00 | 6.43 | 5.65 | **7.09 / 12.77** | 0.459 |
| **median** | **0.90 %** | **0.82 %** | **1.16 %** | **8.67 %** | **3.67 %** | **3.47 %** | | |
| max | 1.36 % | 1.16 % | 2.36 % | 14.0 % | 8.30 % | 8.07 % | | |

(unica is the kernel itself, the other candidates the emulator, which matches the kernel to ≤ 0.015 % on random
data.)

- **Today's kernel against SageAttention 2 over 22 blocks:** median 0.90 % against 1.16 %; SageAttention 2 up to
  2.1× worse (blocks with uneven channels), unica worse in 2 of 22 by ≤ 5 % of their error. A wider and
  stronger result than the 2 blocks of M5.3.
- **Hadamard on today's INT8 Q·Kᵀ: better in all 22 blocks, median 1.12×, at most 1.30×** (step 2, block 17:
  1.36 → 1.05 %). Smaller than the synthetic 3–4× of M4.5. Real channels are uneven by 1.5–2.8 in most blocks
  (×7–13 only in blocks 1, 3 and 33), and unica's error also holds P·V and the bf16 output (0.18 % at block 38),
  which a rotation of Q and K does not touch.
- **FP4 P·V on real data: median 3.67 %, 0.81–8.30 %.** Far kinder than random (13.7 %): real attention is
  sharp, and P's FP4 error shrinks with sharpness (M4.2). Against SageAttention 3 on the same calls: **2.4× less
  error** at the median (3.67 against 8.67 %), at least 1.4× in every block. Against SageAttention 2: 3.2× more.
- **Centring V: median 1.03×, at most 1.47×** (block 38). Real V does have offsets (0.17–0.46 of its RMS), much
  smaller than the 2σ of the synthetic case in M4.4, and the block scales per 16 keys absorb most of them.
- The blocks with very uneven channels (1, 3, 33) are where SageAttention 2 and 3 and FP4 P·V do worst; unica's
  error stays near 1 % there.

## Baselines — two modes, baseline 1 (2026-11-05)

The rule: the baseline is the best result reached so far, and only something better replaces it. A
candidate runs against the current baseline of its mode in the same run, the baseline staying runnable behind
flags. A win becomes the next baseline (git tag `baseline-N`); a loss stays here as an experiment.

| mode | rival | a win |
|---|---|---|
| **accurate** | SageAttention 2 | +1–3 % speed while keeping the error (1.7 % on random data) |
| **fast** | SageAttention 3 | SageAttention 3's speed or better, with less error than it |

A difference in speed below ~2 % is a tie (M3.10: the code layout alone moves ±2 %).

**Baseline 1, both modes:** `attention_int8pv_tma` (INT8 Q·Kᵀ, u8 × s8 P·V, TMA tiles, Σ P̂), K64, 4 warps.
bench/harness.py, whole call, median of 10, two runs per shape in one session, ComfyUI's models unloaded (it kept
~1.9 GB of context):

| | 32 760 tokens | 34 020 tokens (the real call) | rel L2, random |
|---|---|---|---|
| PyTorch SDPA | 518.8 / 523.7 ms | 564.1 / 560.2 ms | 0.23 % |
| SageAttention 2 | 75.2 / 75.5 ms | 81.6 / 81.0 ms | 3.90 % |
| SageAttention 3 | 59.2 / 58.5 ms | 61.9 / 61.2 ms | 19.11 % |
| **UnicaAttention** | **75.1 / 77.2 ms** | **83.3 / 83.3 ms** | **1.73 %** |

Real WAN 2.2, 22 blocks (M4.6), median rel L2: unica 0.90 %, SageAttention 2 1.16 %, SageAttention 3 8.67 %.
A whole 4-step video (M5.5): SageAttention 2 48.7–49.0 s, unica 48.8–49.5 s, SageAttention 3 45.2–45.7 s.

- At 32 760 unica ties SageAttention 2 (75.1 / 77.2 against 75.2 / 75.5 ms). **At the real 34 020 it is 2–3 %
  behind** (83.3 against 81.0–81.6 ms), a gap that M5.4 (81.2 ms, another session) did not show. The accurate
  mode's goal is exactly this gap.
- The fast mode starts 26–36 % behind SageAttention 3 (75–83 against 58–62 ms), with 11× less error on random
  data and 9.6× less on real data.
- From M1 (~770 ms): 10.2× at 32 760.

## M4.7 — hypotheses for both modes (written before any of them is tested)

The approach: many hypotheses, tried one after another. Each with the limit it attacks, a number, and the
test that settles it.

| # | mode | hypothesis | attacks | prediction | test |
|---|---|---|---|---|---|
| H1 | accurate | the 2–3 % gap to SageAttention 2 at 34 020 (Baselines) is mostly noise | — | in alternating rounds in one process, unica / SA2 at 34 020 within ±1 % of the ratio at 32 760. The shape costs only the partial last tile and the last wave, ≤ 0.3 % | harness kernels alternating, both shapes |
| H2 | both | real attention has **negligible tiles**: a block's 64 rows × a 64-key tile in which every P < τ · (running max). Skipping their softmax and P·V changes the output by ≪ the error budget | tensor pipe and the softmax: work that rounds to 0 anyway | very uncertain: **20–40 %** of the (block, tile) pairs at τ = 2⁻¹², error change < 0.05 % abs; 0 % on random data | emulator with a skip threshold and a counter, on the real Q, K, V of a WAN run |
| H3 | fast | FP4 Q·Kᵀ (variant a) on real data is no better than SageAttention 3 | — | real median ≥ 8 %: real attention is sharp, and a's error grows with sharpness (M4.2) | `emu_fp4qk` in the same WAN run |
| H4 | accurate | the layout copies of the ComfyUI node ([b, s, h, d] ↔ [b, h, s, d], M5.1) cost ~2 % of a real call | VRAM traffic | ~1.7 ms of a ~83 ms call (4 × 348 MB at 810 GB/s) | quantize kernels and `store_out` read and write the node's layout; timed in the node |
| H5 | accurate | quantization fused (the K mean and V's max in the same pass as the second read, through L2) and Q quantized inside the attention kernel | VRAM traffic | −1.0 … −1.5 ms of 2.9 ms (M3.20: ~0.8 for K and V, ~0.5 for Q) | compare.py whole call |
| H6 | accurate | Hadamard rotation of Q and K in the quantize pass | Q·Kᵀ error on uneven channels | speed: a tie (< 0.3 ms, the pass is memory-bound); real median 0.90 → ~0.82 % (M4.6) | compare.py + a WAN run |
| H7 | fast | the FP4 P·V kernel | tensor pipe (P·V on OMMA, 2×), FMA pipe (no flush: OMMA accumulates f32 into `out`) | kernel **−12 … −20 %**, 75 → 60–66 ms. The block scales of P come almost free: the max of each 16 keys is a node of the row-max tree already computed, so amax(P) of a block = exp2(block max − m) | the kernel, compare.py, a WAN run |
| H8 | fast | FP4 P·V only in the layers that tolerate it (M4.6: 0.8 % at block 38, 8.3 % at block 1) | — | a per-layer choice keeps the real median near today's at a part of H7's gain | after H7, from the WAN data |

Order: H1 now (local, minutes). H2 and H3 go into one WAN run through the ComfyUI agent. Then H7, the big one.

**H1, result: held — the gap was the harness's order.** Unica and SageAttention 2 in one process, taking turns
(6 rounds, the order flipped every round, median of 5 calls each):

| | unica | SageAttention 2 | unica / SA2, median | per round |
|---|---|---|---|---|
| 32 760 | 76.15 ms | 75.78 ms | 0.996 | 0.970 … 1.017 |
| 34 020 | 81.40 ms | 82.54 ms | **0.988** | 0.965 … 1.002 |

At the real shape unica is level with SageAttention 2 or ~1 % ahead, not 2–3 % behind. bench/harness.py always
runs the kernels in the same order, unica last, after SDPA's half-second calls and SageAttention 3, at the power
limit (M3.22). Its single numbers are fine for a table; a comparison between two kernels needs alternating
rounds. The per-round ratio spreads over ±2–3 %, so the accurate mode's +1–3 % has to be read from medians of
several rounds.

**H2, first look (synthetic):** on random data no tile is ever negligible for a whole block, even at q × 8, where
68 % of single rows' tiles are. A block can skip only when all its 64 rows can, and random rows are independent.
With rows made alike in blocks of 64 (q × 8), 30.5 % of the pairs could skip at τ = 2⁻¹², and the error does not
move (4.059 % both ways). Real video tokens are neighbours in space and time; whether their rows are alike enough
is what the WAN run measures.

**H2 and H3 on real data** (the M4.6 setup again, `measure_every` 7, the same 22 blocks;
`build/wan_attention_errors_m4.7.csv`). Shares of (block, tile) pairs that could skip, by rows deciding together
and threshold τ; errors rel L2:

| step, block | unica | FP4 Q·Kᵀ (a) | FP4 P·V (b) | SA3 | skip, 64 rows, τ = 2⁻¹² / 2⁻²⁰ | skip, 16 rows, τ = 2⁻¹² / 2⁻²⁰ |
|---|---|---|---|---|---|---|
| 1, 1 | 1.108 | 9.16 | 8.30 | 11.95 | 4.3 / 0.0 % | 29.9 / 13.0 % |
| 1, 8 | 0.920 | 7.07 | 5.01 | 8.63 | 2.7 / 0.0 % | 8.3 / 0.0 % |
| 2, 3 | 1.277 | 10.10 | 8.11 | 12.65 | **19.8** / 5.8 % | 37.9 / 12.5 % |
| 3, 33 | 0.876 | 12.43 | 6.43 | 14.00 | **38.6** / 21.1 % | 41.8 / 26.3 % |
| the other 18 blocks | 0.18–1.36 | 0.57–11.46 | 0.81–6.32 | 2.08–11.48 | ≤ 0.7 / 0.0 % | ≤ 4.0 / 0.0 % |
| **median of 22** | **0.90 %** | **7.41 %** | **3.67 %** | **8.67 %** | **0.05** / 0.00 % | 0.86 / 0.00 % |
| mean of 22 | | | | | 3.1 / 1.2 % | 6.1 / 2.4 % |

- **H2: wrong.** Predicted 20–40 % of the pairs at τ = 2⁻¹²; the median block has 0.05 %. Negligible tiles exist
  in 3 of 22 blocks only, the same blocks whose channels are very uneven (M4.6: 1, 3, 33), up to 39 % there.
  Averaged over the blocks, a kernel that skips per block of 64 rows would drop ~3 % of the softmax and P·V work,
  ~1.5 % of the time at best, and it would pay a block-wide vote in every tile. Below what M3.10 showed can be
  read. Per warp (16 rows) there is twice as much, but the per-tile barrier keeps a warp that skips waiting for
  the others. Not pursued.
- The error cost of skipping at τ = 2⁻¹² does not show: `emu_skip12` is below the kernel in every row by
  0.000–0.042 pp, also in the blocks with nothing skipped. That is the emulator's own small offset from the
  kernel, not an effect of skipping (an emulator-to-emulator column would have been the clean comparison).
- **H3: held, nearly.** FP4 Q·Kᵀ on real data: median 7.41 %, against SageAttention 3's 8.67 % (predicted ≥ 8 %).
  FP4 P·V does twice as well (3.67 %) in 21 of 22 blocks; the exception is block 38, where all errors are at the
  floor. Variant b stays the fast mode's path.

## M4.8 — H7: the FP4 P·V kernel, `attention_fp4pv` (prediction, before measuring)

**The m16n8k64 NVFP4 mma, measured on the card first** (a probe with every lane's registers given, against torch):
lane l holds k = 8 · (l % 4) + 32 · (r / 2) + j of row l / 4 + 8 · (r % 2) in nibble j of A register r; k = 8 · (l % 4) +
32 · r + j of column l / 4 in B register r; D as in every m16n8. Scale registers: byte i covers k 16i .. 16i + 15;
with selector 0, lane 4g gives the scales of A's row g, lane 4g + 1 those of row g + 8, lane 4g those of B's column
g, and the other lanes' registers are not read. Random values and scales: 1e-7 from torch (f32 summation order).

**Design** (`attention_int8pv_tma` with the P·V half replaced; Q·Kᵀ, its pipelining, K tiles unchanged):

- **P** against the running max, times 2¹⁰, so block scales of P stay inside e4m3 (≤ 1024 / 6 < 448). No tile-local
  max and no beta: the block scales keep the range, and the mma adds straight into `out` in f32. **No flush.**
- **Block scales of P from the row-max tree.** A lane's 8 scores of a row in one half of the tile and its neighbour's
  (lane ^ 1) are one block of 16 in the mma's k order; lane ^ 2 has the other block. The first two shuffle steps of
  the row max are the block maxes. scale = e4m3(max P / 6), and P̂ = P / scale comes out of the exponent itself:
  2^(s · c + gain − max − lg2(scale)), the same FFMA as before with a per-block offset.
- **V4** (`quantize_v4`): per head and tile, 128 rows of 48 bytes, one per channel: 64 e2m1 in slot order, then the
  e4m3 scales of rows 0..31 placed for 4 LDS.128 per lane. One `ldmatrix.x4` (b16, not transposed) gives B for two
  blocks of 8 channels. A scale per channel outside the mma (max / 2688), applied in `store_out`. Bit for bit the
  emulator (tests/test_fp4.py), after one finding: · (1/6), not / 6, as torch does it (M3.2), or a scale on an e4m3 tie
  (152 between 144 and 160) lands on the other side.
- **Row sum by the tensor core:** a 17th mma per tile against a column of ones gives Σ P̂ · scale, rescaled with
  `out`. No FADD, no IDP4A.
- The emulator follows the kernel exactly (`fp4_slots`, `p_outer`, `v_outer`): kernel vs emulator 0.000–0.031 % on all
  6 shapes of test_attention, cross-attention and a single token included. rel L2 on random data 13.7 % (b of M4.2:
  13.67 %).

Compiled (ptxas -v): **219 registers** (TMA kernel 235), no spills, **40.0 KB** shared memory (46.1).

**Hypotheses** (per warp and tile, against `attention_int8pv_tma`'s 467 executed instructions):

| | predicted |
|---|---|
| removed | 32 IMMA → 17 OMMA; 16 → 8 V `ldmatrix`; flush 64 I2FP + 64 FFMA; 16 F2IP; 8 IDP4A + row sum; beta |
| added | block maxes (~+8), 4 block scales (EX2, cvt to e4m3 and back, LG2: ~24), 16 cvt to e2m1 + packing (~12), 4 LDS.128, shuffle and PRMT of the scale register |
| executed per tile | **~340–360** (−23 … −27 %) |
| tensor time per tile | 64 IMMA → 32 IMMA + 17 OMMA of the same issue time: **−23 %** |
| tensor pipe (ncu) | about as busy as now (~75 %): both the work and the instructions around it shrink |
| kernel time, 40 × 32 760 | **−15 … −22 %**: 73 → 57–62 ms. Under the power limit fewer instructions and half the V bytes per tile mean less energy per tile, so the clock should not drop |
| whole call vs SageAttention 3 (58.5–59.2 ms) | **within ±5 %** |
| error | random 13.7 %; real WAN median 3.7 % (M4.6), against SageAttention 3's 19.1 % and 8.7 % |

## M4.8 — result

ncu, 4 heads (both 1.97 warps per scheduler, 2 blocks per SM):

| | `attention_int8pv_tma` | `attention_fp4pv` | predicted |
|---|---|---|---|
| executed per warp and tile | 467.2 | **369.3 (−21 %)** | 340–360: a little more |
| active cycles | 16.96 M | **14.53 M (−14.3 %)** | |
| tensor pipe | 75.5 % | **67.4 %** | ~75 %: **missed** |
| SFU (xu) | 21.2 % | 28.9 % | |
| FMA / ALU pipe | 10.1 / 12.0 % | 6.5 / 10.7 % | |
| issue slots busy | 34.4 % | 31.5 % | |
| stalls (per warp, K): wait / math pipe / MIO / short scoreboard / barrier | 348 / 344 / 37 / 99 / 136 | 293 / 224 / **70** / 71 / 63 | |
| static SASS of the function (3 copies of the tile) | IMMA 192, F2FP 32, MUFU 114 | IMMA 96 + **OMMA 51**, **F2FP 104**, MUFU 132, SHFL 25 | |

Wall time, whole call, one process, 6 rounds taking turns with the order flipped, ComfyUI's models unloaded:

| | 32 760 | 34 020 |
|---|---|---|
| SageAttention 2 | 80.0 ms | 81.6 ms |
| **SageAttention 3** | **60.6 ms** | **60.8 ms** |
| baseline 1 (`attention_int8pv_tma`) | 78.0 ms | 79.6 ms |
| **`attention_fp4pv`** | **68.6 ms (+13.3 % on SA3)** | **69.8 ms (+14.9 % on SA3)** |

(compare.py in an earlier run, kernel alone: 75.1 → 66.0 ms at 32 760, 78.5 → 68.1 ms at 34 020: −12.1 / −13.2 %.)

| | predicted | measured |
|---|---|---|
| kernel | −15 … −22 % | **−12 … −13 %: missed, below the range** |
| whole call vs SageAttention 3 | within ±5 % | **+13 … +15 %: missed** |

- **What held:** the instructions (−21 %), the registers (219), the error (= the emulator, 13.7 % on random data).
- **What missed:** the tensor time fell by 23 %, the cycles by 14 %: the tensor pipe now idles a third of the time
  (75 → 67 %). The new work sits on the SFU's pipe (F2FP: 16 conversions to e2m1 and 8 for the scales per tile, xu
  21 → 29 %) and on MIO (shuffles for the block maxes and the scale register: MIO stalls ×1.9).
- **Fast mode against SageAttention 3:** 13–15 % slower, with 1.4× less error on random data (13.7 against 19.1 %)
  and 2.4× less on real data (3.7 against 8.7 %, M4.6). Not a win yet by the mode's rule. From baseline 1: −12 %.
- **Under sustained load the gain is smaller.** build/scratch/clock_probe.py (6 s of calls each, kernel alone, two
  pairs): `attention_int8pv_tma` 77.6 / 79.3 ms at 2655 / 2677 MHz, `attention_fp4pv` 71.9 / 72.7 ms at 2715 / 2707
  MHz, all four at 360 W. The FP4 kernel spends less energy per cycle (+1–2 % clock at the same power), and yet
  gains only **−7 … −8 %** here, against −12 … −13 % in rounds of 5 calls. Not explained yet; a 4-step video is
  closer to the sustained case.

## M4.9 — H7a: P's block scales as powers of two (prediction, then result)

**What it attacks:** the chain from a block's max to P's exponents. With e4m3 scales it is EX2 → cvt to e4m3 → cvt
back → LG2 → offset, before the 32 EX2 of P and the P·V mma can start. With 2^e scales e = ceil(s_max · c + 10 −
log2 6 − m) comes from the scores with an F2I, and the scale byte with integer operations.

**Accuracy first** (emulator, `p_pow2`): random 13.74 → **17.17 %**, q × 2 12.23 → 13.87 %, q × 4 10.25 → 10.60 %.

**Hypothesis:** −3 … −6 % of the kernel, if the chain is on the critical path; 0 if ptxas already hides it under
Q·Kᵀ(t + 1).

**Result: 0.** Active cycles 14.51 M against 14.53 M; static MUFU −24 and F2FP −24 (xu 28.9 → 26.2 %), executed
instructions +1.7 % (the integer encoding of the scale byte). Wall time, same run as above: 68.1 / 71.0 ms against
68.6 / 69.8 ms, within the spread. The chain was not the limit: it runs under the next tile's Q·Kᵀ. 3.4 points of
error for nothing; **not kept**, the kernel is back to e4m3 scales (the emulator keeps `p_pow2`).

## M4.10 — the chain to the first P·V mma, measured

**The question:** instruction counts say nothing about the chain. Removing work that runs in parallel with other work
saves nothing (H7a). How long is the way from a tile's start to the first moment its P·V mma can issue, and what
is on it?

**Probe:** `trace_mark_after` (csrc/flash.cu, `UNICA_TRACE=1`) reads `clock64` with a value as its input, so the read
waits until the value exists, wherever ptxas put the instructions that make it. Four points per tile in both TMA
kernels: tile start (after the barrier), the row max known (alpha), P's scale known (beta / the scale register),
P packed (the A fragment: the first P·V mma could issue). SM 0, 1 × 32 760, 24 warps, tiles 2 … n − 2, medians:

| cycles per tile | `attention_int8pv_tma` | `attention_fp4pv` |
|---|---|---|
| whole tile, start to next start | 2808 | 2512 |
| start → row max known | **1397** | **1157** |
| row max → P's scale known | 13 | 25 |
| row max → P packed | 60 | **33** |
| P packed → next tile's start (P·V, barrier) | 1367 | 1280 |

No mark out of order in any tile.

- **The softmax itself is short:** from the max to a packed P, 33 cycles in the FP4 kernel, the scale chain (EX2 →
  e4m3 → back → LG2) 25 of them. That is why H7a gained nothing.
- **The row max is known only after ~1150–1400 cycles**, though the scores of tile t exist when the iteration starts.
  SASS shows why: ptxas spreads the max, its shuffles, the scales and the exponents between the 32 IMMA of
  Q·Kᵀ(t + 1), and a warp issues in order. With two warps sharing the tensor pipe an IMMA occupies a warp ~32
  cycles, so whatever sits between them advances at the IMMA's pace: 32 × ~36 ≈ 1150. After the last IMMA come 17
  MUFU in one run (the rest of P's exponents) and the e2m1 conversions, with no tensor work for this warp.
- So the chain of a tile is the warp's **own stream of tensor instructions**, plus the stretches between them that
  have no mma: the MUFU and F2FP after Q·Kᵀ(t + 1), and in P·V the barrier. The levers this points to: fewer tensor
  instructions per warp (Q·Kᵀ is 32 of the 49 per tile), and moving the work after the last IMMA (17 MUFU, 16 F2FP)
  under the IMMA stream or onto an idle pipe (the FMA pipe is 6.5 % busy).
- k = 64 has a structural cost: one OMMA reads all 64 keys of P, so P·V cannot start before every exponent of the
  tile is done. The u8 path could start its first k32 step after half of them.

## M4.11 — H11: part of P's exponents on the FMA pipe (prediction, then result)

**What it attacks:** the run of 17 MUFU after the last IMMA of Q·Kᵀ(t + 1) (M4.10), the work a warp does there with
no mma, on the SFU's pipe (xu = exactly the 42 MUFU per tile; F2FP runs on the ALU pipe, measured). FlashAttention 4
computes part of its exponents with a polynomial for the same reason.

**Change** (flag `POLY_EXP` = how many of P's 4 A registers, 8 exponents each, use it): `exp2_poly`, x = n + f with n
rounded into the mantissa by adding 1.5 · 2²³, 2^f from a degree-3 near-minimax polynomial on [−0.5, 0.5] (relative
error 7.5e-5), n added to the exponent field as an integer, x clamped at −125. 8 FMA/ALU instructions for one MUFU.

**Hypotheses:** 8 exponents: −1.5 … −2.5 % (the polynomial fills the issue slots a warp leaves between MUFU that wait
for the SFU); 16: −1 … +2 %. Error: unchanged against FP32, < 0.1 % from the emulator.

**Result** (compare.py, 5 rounds, kernel alone; ncu, 4 heads):

| exponents on the polynomial | 0 | 8 | 16 | 32 |
|---|---|---|---|---|
| kernel | 64.3 ms | 65.0 ms (+1 %) | 67.1 ms (+4 %) | 73.4 ms (+14 %) |
| active cycles | 14.51 M | 14.76 M (+1.7 %) | 15.10 M (+4.1 %) | |
| executed per warp and tile | 368.7 | 418.1 | 472.9 | |
| xu / FMA pipe | 28.9 / 6.5 % | 23.0 / 11.2 % | 17.2 / 15.8 % | |
| tensor pipe | 67.5 % | 66.4 % | 64.9 % | |
| rel L2 vs FP32 (4 × 4096) / vs the emulator | 13.736 % / 0.04 % | 13.736 % / 0.18 % | 13.736 % / 0.25 % | 13.736 % / 0.35 % |

- **Wrong, in the wrong direction.** The SFU's pipe emptied as planned (29 → 17 %), and each added instruction cost
  ~0.4 cycles while the removed MUFU saved nothing. The run of MUFU is not on the critical path either. Not kept.
- What the numbers do say: per scheduler a warp's tile takes 1162 cycles, of which its 49 tensor instructions take
  794 (16.2 each, the peak spacing): **68 %**, as ncu reads. The other 368 cycles per tile no tensor instruction of
  either warp on the scheduler issues: both warps are between mma at the same time. Neither the softmax chain (M4.9,
  M4.10) nor the SFU (M4.11) is what holds them there.
- The lever left on this kernel's own terms is the number of tensor instructions per warp: Q·Kᵀ is 32 of the 49.
  That is H9.

## M4.12 — H9, step 1: Q·Kᵀ in NVFP4 too, `attention_fp4` (prediction, before measuring)

**Design.** Q and K in NVFP4 along d (e2m1, an e4m3 scale per 16 channels, one float scale per head; K minus its mean
as before). m16n8k64 sums over d: a tile takes 2 k-steps × 8 blocks of keys = **16 OMMA instead of 32 IMMA**. K4 / Q4
rows of 80 bytes: 64 bytes of e2m1 in channel order, then the 8 scales; one `ldmatrix.x4` gives a block's B registers
of both k-steps and one LDS.64 its two scale registers. Q stays in registers (8 + 2). The scores come out in the same
m16n8 layout, so softmax and P·V are `attention_fp4pv`'s. The two float scales per head fold into scale_log2.
`quantize_fp4_rows` is the emulator bit for bit; the kernel is the emulator (`attention_fp4pv`'s configuration with
`qk="nvfp4"`) within 0.05 % on the 6 shapes, cross-attention included. Compiled: 227 registers, no spills,
**22.5 KB** shared memory (fp4pv 40.0, baseline 46.1). Per-head max of K − mean still costs an extra pass in torch.

**Hypotheses** (against `attention_fp4pv`):

| | predicted |
|---|---|
| tensor instructions per warp and tile | 49 → 33 (−33 %) |
| active cycles | **−22 … −33 %**: M4.11 split a warp's tile per scheduler into 794 cycles of tensor work and 368 without. −22 % if those 368 stay, −33 % if they shrink with the work |
| kernel, 40 × 32 760 | 64 → **43–50 ms** |
| whole call against SageAttention 3 (60.6 ms) | **10–22 % faster** |
| error | the emulator: 19.6 % on random data (SageAttention 3 19.1 %); real WAN from the M4.12 run |

**Real WAN, the M4.6 setup again** (`build/wan_attention_errors_m4.12.csv`, 22 blocks; `emu_fp4pv` now the kernel's
exact configuration, so it reads 3.81 % where M4.6's variant b read 3.67 %):

| step, block | unica | SA2 | SA3 | FP4 P·V | FP4 Q·Kᵀ alone | **FP4 both** | both / SA3 |
|---|---|---|---|---|---|---|---|
| 0, 6 | 0.662 | 0.708 | 5.50 | 2.54 | 5.15 | 5.60 | 1.02 |
| 0, 13 | 1.111 | 1.062 | 8.70 | 3.76 | 8.48 | 9.19 | 1.06 |
| 0, 20 | 1.151 | 1.569 | 9.36 | 5.74 | 7.76 | 9.32 | 1.00 |
| 0, 27 | 0.843 | 0.873 | 7.46 | 3.83 | 6.10 | 6.93 | 0.93 |
| 0, 34 | 0.764 | 1.001 | 7.12 | 3.58 | 5.84 | 6.67 | 0.94 |
| 1, 1 | 1.108 | 2.326 | 11.95 | 8.27 | 9.16 | 12.02 | 1.01 |
| 1, 8 | 0.920 | 1.584 | 8.63 | 5.07 | 7.07 | 8.61 | 1.00 |
| 1, 15 | 1.076 | 1.501 | 9.19 | 4.53 | 8.41 | 9.52 | 1.04 |
| 1, 22 | 1.231 | 1.736 | 10.36 | 5.35 | 9.34 | 10.71 | 1.03 |
| 1, 29 | 0.583 | 0.781 | 5.70 | 2.56 | 4.76 | 5.37 | 0.94 |
| 1, 36 | 0.352 | 0.437 | 3.46 | 1.64 | 2.83 | 3.15 | 0.91 |
| 2, 3 | 1.277 | 2.361 | 12.65 | 8.12 | 10.10 | 12.78 | 1.01 |
| 2, 10 | 0.658 | 0.836 | 5.76 | 2.68 | 6.20 | 6.33 | 1.10 |
| 2, 17 | 1.361 | 1.662 | 11.44 | 4.36 | 11.46 | 12.22 | 1.07 |
| 2, 24 | 0.760 | 0.943 | 6.69 | 3.25 | 5.80 | 6.50 | 0.97 |
| 2, 31 | 1.012 | 1.251 | 9.45 | 3.79 | 9.22 | 9.96 | 1.05 |
| 2, 38 | 0.176 | 0.180 | 2.08 | 0.79 | 0.57 | **0.96** | **0.46** |
| 3, 5 | 0.684 | 0.851 | 6.35 | 2.89 | 5.58 | 6.18 | 0.97 |
| 3, 12 | 0.629 | 0.595 | 4.95 | 2.60 | 4.27 | 4.77 | 0.96 |
| 3, 19 | 1.263 | 1.674 | 11.18 | 4.88 | 11.07 | 12.08 | 1.08 |
| 3, 26 | 1.353 | 2.068 | 11.48 | 6.48 | 10.03 | 11.76 | 1.02 |
| 3, 33 | 0.876 | 1.830 | 14.00 | 6.44 | 12.43 | 13.94 | 1.00 |
| **median** | **0.90 %** | **1.16 %** | **8.67 %** | **3.81 %** | 7.41 % | **8.90 %** | |

- **FP4 in both halves is SageAttention 3's class on real data too:** median 8.90 against 8.67 %, below it in 11 of
  22 blocks, the ratio 0.91–1.10 everywhere but the last block (0.46). The error comes from Q·Kᵀ (7.41 % alone).
- So `attention_fp4` alone cannot meet the fast mode's rule (less error than SageAttention 3). Whatever it gains in
  speed has to be mixed with `attention_fp4pv` per layer, or bought back elsewhere.

## M4.12 — result: the Q·Kᵀ half got faster, and the tile now waits for its data

Whole call, one process, 6 rounds taking turns with the order flipped, ComfyUI's models unloaded:

| | 32 760 | 34 020 |
|---|---|---|
| SageAttention 3 | 58.0 ms | 62.8 ms |
| `attention_fp4pv` | 66.3 ms | 73.3 ms |
| **`attention_fp4`** | **72.8 ms** | **78.4 ms** |
| baseline 1 | 76.2 ms | 82.4 ms |
| SageAttention 2 | 75.8 ms | 83.5 ms |

compare.py, 5 rounds: `attention_fp4pv` kernel 64.5 ms + quantization 2.94 ms; **`attention_fp4` kernel 62.5 ms (−3 %) +
quantization 10.35 ms**. The quantization in torch (a float copy of K − mean for its max, a max over Q) is a first
version's cost and can go into the quantize kernels; the kernel is the finding.

| | `attention_fp4pv` | `attention_fp4` | predicted |
|---|---|---|---|
| kernel, compare.py | 64.5 ms | 62.5 ms (−3 %) | 43–50 ms: **missed by far** |
| active cycles (ncu, 4 heads) | 14.49 M | 15.06 M (+4 %) | −22 … −33 % |
| executed instructions | 1.553 G | 1.452 G (−6.5 %) | |
| tensor pipe | 67.6 % | **43.8 %** | |
| stalls (per warp, K): long scoreboard / barrier / math pipe | 124 / 66 / 224 | **383** / 20 / 95 | |
| trace, SM 0: whole tile | 2490 | 2638 | |
| trace: start → P packed | 1199 | **672 (−44 %)** | |
| trace: P packed → next tile's start | 1236 | **1883 (+650)** | |

- The Q·Kᵀ half did what it should: the way to a packed P fell from 1199 to 672 cycles.
- **All of it went into waiting at the end of the tile**, where a warp waits on the mbarrier for the next tile's K
  and V (long scoreboard ×3). The copies are issued one iteration ahead, ~2500 cycles (~1 µs). That covered the
  L2 → shared memory latency while a tile took that long; a shorter tile now outruns its data.
- Next (H9b): a deeper pipeline, copies issued 2–3 tiles ahead. The FP4 tiles are small (K4 5 KB, V4 6 KB), so a third
  and fourth stage cost 11 KB each: 34 or 45 KB per block, still 2 blocks per SM.

## M4.13 — H9b: copies 2–3 tiles ahead (prediction, before measuring)

**Change** (`STAGES`, default 2 = M4.12, `attention_fp4` only): STAGES buffers of K and of V; iteration j reads K(j + 1)
and V(j), one group of copies on `full[j % STAGES]`; iteration t issues group t + STAGES − 1. Why no buffer is
overwritten early is in the kernel's comment: the slots a group writes and the barrier it uses were last read in
iteration t − 1, before the barrier at the top of iteration t. Output bit for bit the same at 2, 3 and 4 stages on
7 shapes (1 to 64 tiles, cross-attention, a single token). Compiled: 2 stages 227 registers, 22.6 KB; 3 stages 237,
33.8 KB; 4 stages 237, 45.1 KB; no spills, 2 blocks per SM in all.

**Hypotheses:**

| | 3 stages | 4 stages |
|---|---|---|
| trace: P packed → next tile's start | 1883 → **~1250** (the wait gone, P·V and the barrier as in fp4pv) | the same |
| long-scoreboard stalls | 383 K → ~100 K | the same |
| kernel, 40 × 32 760 | 62.5 → **47–50 ms (−20 … −25 %)** | within 2 % of 3 stages: one iteration ahead more buys nothing once the latency is covered |
| whole call against SageAttention 3 | kernel faster than SageAttention 3's whole call; the whole call still carries 10 ms of torch quantization | |

**Result: 0, wrong.** compare.py, 5 rounds, kernel alone: 2 stages 62.6 ms, 3 stages 62.5, 4 stages 62.5. ncu: active
cycles 15.02 M → 15.04 M, long-scoreboard stalls 373 K → 369 K. The data was not late for want of a head start.

**Where the warps wait** (ncu SourceCounters, 622 K stall samples, 2 stages): **~37 % on the mbarrier wait loop**
(`SYNCS.PHASECHK.TRANS64.TRYWAIT` and its back branch, both barriers), OMMA 19 %, MUFU 7.5 %, FMNMX 5.7 %. So the
tiles do arrive late, and a deeper pipeline does not help: the copies are slow, not early enough.

| (ncu, 4 heads) | `attention_int8pv_tma` | `attention_fp4pv` | `attention_fp4` |
|---|---|---|---|
| TMA box rows: K / V | 64 × 128 B / 32 × 256 B | 64 × 128 B / **128 × 48 B** | **64 × 80 B / 128 × 48 B** |
| bytes from L2 | 20.4 GB | | 15.0 GB |
| long-scoreboard stalls (per warp, K) | **22** | 124 | **373** |

The FP4 kernel moves a quarter less data and waits 17 times longer for it, and the waits grow with the short rows.
A 2D box of many short rows (48 and 80 bytes, not on 32-byte sector boundaries) is a slow way to copy a tile.

## M4.14 — H9c: each tile as one contiguous block (prediction, before measuring)

**Change** (`BULK_COPY`, `attention_fp4`): a K4 tile (64 × 80 = 5120 bytes) and a V4 tile (128 × 48 = 6144 bytes) each lie in
one piece in global memory, once K4 is padded to whole tiles per head (zeros), as V4 already is. `cp.async.bulk`
(1D, the same mbarrier accounting) copies them with no box at all; the layout in shared memory is the same. Output
bit for bit the same with and without, at 2 and 3 stages, on 6 shapes.

**Hypotheses:** long-scoreboard stalls down to the baseline's order (~20–120 K); tensor pipe 44 → ~65 %; kernel 62.5 →
**45–50 ms** (−20 … −28 %); and the same change on `attention_fp4pv`'s V4 later, a few percent.

**Result: held.**

| | 2D box | 1D bulk | 1D bulk, 3 stages | predicted |
|---|---|---|---|---|
| kernel, compare.py (5 rounds, one run) | 64.8 ms | **45.9 ms (−29 %)** | 47.6 ms | 45–50 ms: **held** |
| whole call (with 10.6 ms of torch quantization) | 76.2 ms | 56.5 ms | 58.6 ms | |
| active cycles (ncu, 4 heads) | 15.10 M | **10.84 M (−28 %)** | | |
| long-scoreboard stalls (per warp, K) | 378 | **31** | | ~20–120: **held** |
| tensor pipe | 43.7 % | **60.9 %** | | ~65 %: close |
| bytes from L2 | 15.2 GB | **12.1 GB** (11.8 expected: no more partial sectors) | | |
| executed instructions | 1.437 G | 1.292 G (fewer rounds of the wait loop) | | |
| barrier stalls | 21 K | 102 K | | |
| sustained, 6 s of calls (clock_probe) | | **47.5 ms at 2715 MHz, 360 W** (`attention_fp4pv` 65.4 ms at 2640) | | |

- **The tiles were slow to copy, not late to start.** A 2D TMA box of 64 rows of 80 bytes and 128 of 48, none on a
  32-byte sector boundary, read 1.27× the bytes and kept the warps on the mbarrier. One contiguous `cp.async.bulk` per
  tile reads exactly the tile.
- A third stage adds nothing (47.6 against 45.9 ms, within the spread): two are enough once the copy is fast.
- **The kernel alone (45.9 ms) is now faster than SageAttention 3's whole call (58–60 ms).** The whole call still
  carries 10.6 ms of quantization in torch, the next thing to go. Error: SageAttention 3's class (M4.12 real data).
- The waits moved to the per-tile barrier (21 → 102 K): with the copies fast, the warps of a block now wait for each
  other.

## M4.15 — the same copy for `attention_fp4pv`, and the FP4 quantization without the float copy (prediction)

- `attention_fp4pv` takes V4 tiles by `cp.async.bulk` too (`BULK_COPY`; its K stays an int8 box of 128-byte rows).
  Output bit for bit the same. **Prediction:** long-scoreboard stalls 124 → ~30 K, kernel **−5 … −10 %** (64 → 58–61 ms).
- `quantize_fp4_rows`: the head's max |x − mean| from each channel's min and max (`torch.aminmax`, one pass), instead of
  a float copy of K − mean and a max over it: max |x − mean| = max over channels of max(max_c − mean_c, mean_c −
  min_c), the extreme elements going through the same float subtraction, so the scale is the same bit for bit (the
  quantizer test passes unchanged). **Prediction:** `attention_fp4`'s quantization 10.6 → **~3.3 ms** (Q: aminmax +
  quantize; K: mean + aminmax + quantize; V: aminmax + quantize; ~2.6 GB at 810 GB/s), whole call ~46 + 3.3 ≈ **49 ms,
  15–18 % below SageAttention 3**.

**Result** (compare.py, 5 rounds, one run):

| | whole call | quantization | kernel | predicted |
|---|---|---|---|---|
| baseline 1 | 77.8 ms | 2.90 ms | 74.9 ms | |
| `attention_fp4pv`, 2D box | 70.5 ms | 2.98 ms | 67.8 ms | |
| `attention_fp4pv`, bulk V | 66.2 ms | 2.96 ms | **62.8 ms (−7.4 %)** | −5 … −10 %: **held** |
| `attention_fp4`, bulk K and V | **50.7 ms** | **3.82 ms** (was 10.6) | 47.2 ms | ~3.3 ms quantization: close |

**Against SageAttention 3, whole call** (one process, 6 rounds taking turns, order flipped; the card ran fast this
time, all kernels with it):

| | 32 760 | vs SA3 | 34 020 | vs SA3 |
|---|---|---|---|---|
| SageAttention 2 | 69.5 ms | +29.4 % | 78.8 ms | +33.0 % |
| **SageAttention 3** | **53.7 ms** | | **59.2 ms** | |
| baseline 1 | 72.7 ms | +35.6 % | 80.9 ms | +36.6 % |
| `attention_fp4pv` (bulk) | 61.0 ms | +13.7 % | 68.2 ms | +15.2 % |
| **`attention_fp4` (bulk)** | **46.5 ms** | **−13.3 %** | **53.6 ms** | **−9.6 %** |

Predicted −15 … −18 %: the direction held, the size a little short. **`attention_fp4` is faster than SageAttention 3**
at its error class (random 19.2 against 19.1 %, real WAN median 8.90 against 8.67 %). From M1 (~770 ms): 16.6×.
`BULK_COPY=1` is now the default in both FP4 kernels (the 2D box stays behind `BULK_COPY=0`).

By the fast mode's rule this is not yet a win: the speed is there, the lower error is not. Two ways from here: lower
`attention_fp4`'s error at its speed, or mix it per layer with `attention_fp4pv` (3.8 % real, +14 % time).

## M4.16 — what surrounds the kernel in a real WAN call (measured before any fusion)

The question: can RoPE be fused into our quantization, and what would it save? ComfyUI 0.42's `WanSelfAttention`
computes `q = norm_q(q(x))`, then `apply_rope1(q, freqs)` (comfy_kitchen, CUDA backend), the same for k, and hands q, k, v
to `optimized_attention` in [b, s, heads, d]. Our node then copies them to [b, heads, s, d] and the output back.

Measured in ComfyUI's own environment (`.venv-13`, torch 2.14), real shape 1 × 34 020 × 40 × 128 (one tensor 348 MB),
real positions (21 × 30 × 54) and the model's `EmbedND` [44, 42, 42], median of 20 (scratchpad rope_overhead.py):

| | per call |
|---|---|
| RMSNorm over 5120, one of q / k | 0.87 ms |
| RoPE, `apply_rope1` (comfy_kitchen), one of q / k | 0.94 ms |
| the node's layout copy in, one of q / k / v | 0.89 ms |
| the node's layout copy out | 0.90 ms |
| quantization: baseline / `attention_fp4pv` / `attention_fp4` | 3.06 / 3.12 / 3.97 ms |
| kernel: baseline / `attention_fp4pv` / `attention_fp4` | 78.8 / 65.0 / 48.2 ms |

- **Found on the way:** the node's `x.transpose(1, 2).to(torch.bfloat16, memory_format=torch.contiguous_format)` copies
  nothing when the dtype already matches (`is_contiguous()` is False after it). The copy happens later, inside
  `unica.attention`'s `.contiguous()`, so it was counted as quantization in a first version of this measurement
  (5.7 instead of 3.1 ms). The cost is the same, its place is different.
- **Removable without touching the model** (H4: quantize kernels that read [b, s, heads, d] directly, `store_out`
  writing it): 3 × 0.89 + 0.90 = **3.56 ms** per call.
- **Removable by fusing RoPE into the quantization** (needs q and k before `apply_rope1`, so a patch of
  `WanSelfAttention.forward`): 2 × 0.94 = **1.88 ms**. RoPE in float inside our kernel also skips one rounding of the
  rotated q and k to bf16.
- **And RMSNorm** (the same patch; a reduction over all 40 heads of a token, before the per-head work): 2 × 0.87 =
  **1.74 ms**.

| per call, around and in the kernel | baseline | `attention_fp4pv` | `attention_fp4` |
|---|---|---|---|
| RoPE + copies + quantization + kernel | 87.3 ms | 73.5 ms | 57.6 ms |
| copies (H4) | 4.1 % | 4.8 % | **6.2 %** |
| + RoPE | 6.2 % | 7.4 % | **9.4 %** |
| + RMSNorm | 8.2 % | 9.8 % | **12.4 %** |

Over a 4-step video (160 self-attention calls) that is 0.57 s for the copies, 0.87 s with RoPE, 1.15 s with RMSNorm, of
~45–49 s: 1–2.5 %. The attention call itself gets 6–12 % faster.

## M4.17 — step 1: the caller's layout, no copies (prediction, before measuring)

**Change:** `TokenLayout` (csrc/flash.cu, `unica.TokenLayout`): the strides of a [b, h, s, d] view and its heads, passed by
value. The quantizers (`quantize_int8`, `quantize_v`, `quantize_v4`, `quantize_fp4_rows`) read q, k, v through it, and
the three fast kernels (`attention_int8pv_tma`, `attention_fp4pv`, `attention_fp4`) write their output through it, so a
[b, s, h, d] tensor seen through a transpose is read and written in place. `unica.attention` keeps such tensors as they
are (`readable`: last dimension contiguous, tokens on 16-byte boundaries), and `empty_like` gives the output q's layout.
The other kernels still get contiguous copies. The node's `unica_nhd` drops its copies. Bit for bit the same output as
the contiguous inputs, all shapes and the three kernels (`test_callers_layout_changes_nothing_but_copies`); registers
unchanged.

**Hypothesis:** per real call (1 × 34 020 × 40 × 128, through `unica_nhd`) **−3.56 ms**, the three copies in and the one out
of M4.16: baseline 87.3 → ~83.7 ms (−4.1 %), `attention_fp4` 57.6 → ~54.0 ms (−6.2 %).

**Result: held.** Input in ComfyUI's layout, the node's path before and after, one process, 6 rounds taking turns:

| | with the copies | in place | | predicted |
|---|---|---|---|---|
| baseline (`attention_int8pv_tma`) | 85.78 ms | **82.36 ms** | **−3.42 ms (−4.0 %)** | −3.56 ms (−4.1 %) |
| `attention_fp4` | 56.95 ms | **53.02 ms** | **−3.93 ms (−6.9 %)** | −3.56 ms (−6.2 %) |

The quantizers read through the transpose as fast as from contiguous memory (a token's 128 channels are still one
256-byte run). Next on this path: RoPE inside the quantization (1.9 ms), then RMSNorm (1.7 ms).

## M4.18 — step 2: RoPE inside the quantization (prediction, before measuring)

**Change.** `unica.attention(q, k, v, rope=freqs)`: q and k come before RoPE, `freqs` is ComfyUI's table (float32 [b or 1,
s, 1, 64, 2, 2], every rope option of the model already in it). `quantize_int8` and `quantize_fp4_rows` rotate each
8-channel chunk in float on the way in (`rope_chunk`: two rounded products, then the sum, as `apply_rope1`), then
subtract the mean. The K mean and the FP4 per-head scales need the rotated values first: `rope_stats`, one pass that
gives Σ, min and max per channel of the rotated x (block sums added in a fixed order, deterministic). It replaces
`k.mean` (and, for `attention_fp4`, `aminmax`), so no pass is added. The rotated q and k are never written, nor rounded
to bf16. `bench/emulate.py` got `rope()` (the same order of operations) and `wan_rope_table()`. tests/test_rope.py:
`rope_stats`' min and max exact, its mean within 1e-6; `quantize_int8` and `quantize_fp4_rows` with `rope=` bit for bit
the emulator's quantization of rope(x) − mean; the three fast kernels with `rope=` within 0.006 % of the emulator on the
rotated q and k (189 tokens, batch 1 and 2).

**Hypotheses** (real shape, ComfyUI's environment; before: `apply_rope1` on q and k, then the kernel in place; after:
`rope=`): **−1.9 ms** per call for the baseline (the read and write of RoPE for q and k), **−2.3 ms** for `attention_fp4`
(also `k.mean` and `aminmax` merged into one pass of statistics). Error against FP32 on the rotated q and k: equal or a
little lower, the bf16 rounding of the rotated q and k being gone.

**Result** (ComfyUI's `.venv-13`, real shape, 6 rounds taking turns): `wan_rope_table` equals ComfyUI's `EmbedND` table
bit for bit; `rope()` differs from comfy_kitchen's `apply_rope1` by at most 2.6e-3 of the largest value, the size of
its bf16 rounding (the pairing is right: a wrong one would differ by O(1)).

| | `apply_rope1` + kernel | `rope=` | | predicted | rel L2 vs FP32 (4 heads) |
|---|---|---|---|---|---|
| baseline | 84.29 ms | 85.62 ms | **+1.32 ms** | −1.9 ms: **wrong** | 1.766 → 1.750 %: held |
| `attention_fp4` | 55.28 ms | 55.86 ms | **+0.58 ms** | −2.3 ms: **wrong** | 19.468 → 19.474 % |

**Why it lost:** the table holds a float 2 × 2 matrix per pair of channels, 8 bytes per element of q, four times q itself.
It is the same for all 40 heads of a token, and every thread of every head reads it again: ~1.4 GB from L2 per pass,
three passes with RoPE (q's quantization, k's statistics, k's quantization), ~4.2 GB, ~1.7 ms at the ~2.5 TB/s L2 rate
of M3.17. That is the whole gain and more.

## M4.19 — RoPE from a compact table (prediction, before measuring)

**Change.** WAN's matrices are rotations, [[c, −s], [s, c]] (ComfyUI's `rope()` stacks cos, −sin, sin, cos). The kernels
take (c, s) per pair in fp16 instead: 2 bytes per element of q, a quarter of the table, one 16-byte load per 8-channel
chunk. `unica.rope_table` checks that the table is a rotation, converts it once and keeps it while the same table comes
back (all 40 blocks of a step share it). fp16's error on c and s, ≤ 2⁻¹², is 8 times below the bf16 rounding of the
rotated q and k that the fusion removes. The emulator rotates with the same fp16 (c, s) and the same order of
operations, so the tests stay bit for bit.

**Hypothesis:** L2 traffic for the table ~0.35 GB per pass, ~0.14 ms; three passes ~0.4 ms. Per call: baseline **−1.4 ms**
(1.9 − 0.4 − the rest), `attention_fp4` **−1.8 ms**.

**Result: held, a little better than predicted.** Same measurement as M4.18:

| | `apply_rope1` + kernel | `rope=`, compact table | | predicted | rel L2 vs FP32 (4 heads) |
|---|---|---|---|---|---|
| baseline | 85.12 ms | **82.83 ms** | **−2.29 ms (−2.7 %)** | −1.4 ms | 1.766 → 1.750 % |
| `attention_fp4` | 55.20 ms | **53.16 ms** | **−2.04 ms (−3.7 %)** | −1.8 ms | 19.468 → 19.471 % |

tests/test_rope.py bit for bit again with the fp16 (c, s) in the emulator. The compact table is kept by a weak reference
to ComfyUI's table, not by its address: ComfyUI builds a new table every step, and the allocator may hand it the
old one's address. (The FP32 reference here rotates with the same fp16 (c, s); against ComfyUI's float table it differs
by ≤ 2⁻¹² per rotation.) Still to do for step 2: the node, which has to take q and k before `apply_rope1`.

**The node** (`Unica Attention (native)`, optional input `fuse_rope`, default off; optional so that saved workflows
still validate): `fused_rope_forward` replaces `WanSelfAttention.forward` on each block, as an object patch of the
model clone only (`add_object_patch`, nothing global). It is the original forward of ComfyUI 0.42 with q and k handed to
unica before RoPE, together with the model's own table; an `attn1_patch` (it wants the rotated q and k), another head
size or a table that does not fit go to the original forward. Checked with ComfyUI's real class (2 heads, 360 tokens,
random weights, scratchpad fused_forward_check.py): fused against the original forward, both on unica, 0.24 % apart
(the bf16 rounding of the rotated q and k, now gone); against ComfyUI's own SDPA 0.293 % (fused) and 0.295 % (original);
with an `attn1_patch` it returns the original's output bit for bit. A video A/B is with the ComfyUI app's agent.

**Found by the first video run:** `fuse_rope` crashed on the first call: `Inference tensors do not track version counter`
in `compact_rope`. ComfyUI samples under `torch.inference_mode`, where the table has no `_version`; every test had run
outside it. The cache now checks the version only for tensors that have one (the weak reference still tells a new
table from the kept one), and `test_under_inference_mode` runs the path the way ComfyUI does.

**Video A/B** (the Rigsy workflow, the baseline kernel, seed 42, same prompt, models kept loaded; runs taking turns, a
warm-up with `fuse_rope` on not counted; both variants already without the layout copies of M4.17):

| | run 1 | run 2 | mean |
|---|---|---|---|
| `fuse_rope` off | 47.2 s | 47.9 s | 47.55 s |
| `fuse_rope` on | 46.4 s | 47.0 s | **46.70 s (−0.85 s, −1.8 %)** |

- Predicted per video: 160 calls × 2.3 ms ≈ **−0.37 s**. Measured −0.85 s: both "on" runs below both "off" runs, but the
  spread inside a variant (0.6–0.7 s) is as large as the difference. The direction holds; the size needs more rounds.
- "fuse_rope: 40 self-attention blocks patched" twice per run (high- and low-noise models), no warning or error.
- **The two videos are both clean and not the same**: from the middle on the motion takes another path (frame 40: a wave
  in one, a hand in the hair in the other), PSNR 24.0 dB over the video. Each variant is deterministic (two runs of each
  byte for byte equal). RoPE in float instead of bf16 changes the rounding of q and k, and 4 steps carry that into a
  different trajectory, as with SageAttention 2 / 3 / unica in M5.5. A wrong rotation (axis or frequency) would break
  the motion or the image; neither happens.

## M4.20 — the fast mode's error: Q smoothing (B) and a per-layer map (A), measured before any kernel

**B, Q smoothing.** q = q̄ + (q − q̄): quantize q − q̄, add q̄ · k back. Unlike K's mean, q̄ · k varies along the keys,
so softmax does not cancel it. SageAttention 3 takes q̄ per block of 128 queries and a precomputed [N/128 × N] matrix
of corrections (~1.5 GB per call here: our speed margin). One q̄ per head makes the correction one vector per key,
b_j = q̄ · k_j: a GEMV before the kernel, one addition per score or folded into the K tile. Emulator (`smooth_q`: "head"
or a block size; q̄ · k in float on the unquantized k), `attention_fp4`'s configuration, 4 × 4096:

| q | FP4 both | + q̄ per head | + q̄ per 64 queries |
|---|---|---|---|
| random | 19.63 % | 19.65 % | 19.56 % |
| + 1σ offset per channel | 23.81 % | **19.37 %** | 19.17 % |
| + 3σ offset per channel | 24.02 % | **12.15 %** | 12.18 % |

Nothing on random data, by construction; a lot on channel offsets; and per head does as well as per 64 queries.

**A, a per-layer map.** M4.15's oracle mix (FP4 both where its error stays below SageAttention 3's, FP4 P·V elsewhere)
reached SageAttention 3's speed at 5.4 % against 8.7 % real error. A real policy needs a map fixed in advance, which only
works if a block's tolerance does not depend on the seed and the prompt.

**Run** (ComfyUI app's agent): the M4.6 setup twice, seed 42 with the usual prompt and seed 7 with another one, both
`measure_every` 7 (the same 22 (step, block) pairs); new columns `emu_fp4_both_smoothq` (per head), `_smoothq64`,
`emu_today_smoothq` (the INT8 baseline with q̄ per head), `q_offset` (RMS of Q's channel means over RMS of Q).

**Hypotheses:** real Q carries channel offsets like V's (`q_offset` 0.2–0.4); q̄ per head takes FP4 both's real median
8.9 → **~7–8 %**, still above FP4 P·V (3.8 %); per 64 queries within 0.3 points of per head. The INT8 baseline barely
moves (its scale is per 128 tokens over all of d). The block ranking of FP4 both's error holds across the two runs
(rank correlation > 0.8).

**Result** (`build/wan_attention_errors_m4.20.csv`; A: seed 42, the usual prompt, 22 rows; B: seed 7, "a red fox walking
through fresh snow…" over the same input image, 23 rows). Medians:

| | A | B |
|---|---|---|
| `q_offset` | 0.40 (0.21–0.50) | 0.42 (0.27–0.61) |
| FP4 both | 8.90 % | 7.15 % |
| + q̄ per head | 8.36 % (−6 %) | 6.92 % (−3 %) |
| **+ q̄ per 64 queries** | **6.92 % (−22 %)** | **6.14 % (−14 %)** |
| SageAttention 3 | 8.67 % | 7.28 % |
| FP4 both + q̄ per head below SageAttention 3 | 18 of 22 blocks (11 without) | 20 of 23 |
| baseline (INT8) → + q̄ per head | 0.898 → 0.862 % | 0.844 → 0.820 % |
| FP4 P·V | 3.81 % | 3.58 % |

- Real Q does carry channel offsets (0.40–0.42, the top of the predicted range), but **they move along the tokens**: one
  q̄ per head takes only 3–6 % off (predicted ~7–8 % absolute: A's 8.36 % just misses), one per 64 queries 14–22 %. On
  synthetic data, with one offset for all tokens, the two were equal.
- With q̄ per 64 queries FP4 both sits 20 % below SageAttention 3's median, the fast mode's rule, but its correction is a
  [N/64 × N] matrix, ~1.5 GB per call.
- The baseline gains 3–4 % from q̄ per head for the cost of one vector per key: a free candidate for the accurate mode.
- **A bug in the measurement:** the node's call count ran on from A into B, so B sampled other blocks (calls ≡ 1 mod 7,
  not 0) and A's per-layer map cannot be compared block by block. The count is now step · blocks + block + 1 within a
  run (the step advances when sigma falls, restarts when it rises), and the CSV has `step` and `block`.

**Next run:** q̄ per 128 queries (SageAttention 3's block), 512 and per frame (1620 tokens: WAN's tokens run frame by
frame, 21 × 30 × 54). If q̄ moves mostly between frames, per frame keeps most of the 64-query gain with a correction of
21 rows per head (~57 MB per call). A and B again, with the fixed count, for the map.

**M4.20b result** (`build/wan_attention_errors_m4.20b.csv`; A and B on the same 22 (step, block) pairs, the per-run count
checked; medians, and the blocks below SageAttention 3 out of 22):

| | A | B | below SA3, A / B |
|---|---|---|---|
| SageAttention 3 | 8.67 % | 8.89 % | |
| FP4 both | 8.90 % | 9.08 % | 11 / 10 |
| + q̄ per head | 8.36 % | 8.49 % | 18 / 20 |
| **+ q̄ per frame (1620 queries)** | **7.77 %** | **8.09 %** | **22 / 22** |
| + q̄ per 512 | 7.49 % | 7.85 % | 22 / 22 |
| + q̄ per 128 (SageAttention 3's block) | 7.05 % | 7.49 % | 22 / 22 |
| + q̄ per 64 | 6.92 % | 7.31 % | 22 / 22 |
| FP4 P·V | 3.81 % | 4.03 % | 22 / 22 |
| baseline (INT8) → + q̄ per head | 0.898 → 0.862 % | 0.915 → 0.874 % | |

- **Per frame keeps about half of the 64-query gain** (−13 / −11 % against −22 / −20 %) and already puts FP4 both below
  SageAttention 3 **in every block of both scenes**, median −10 / −9 %. Its correction is 21 rows per head.
- **The per-layer map is stable:** rank correlation between the two scenes 0.98 for FP4 both's error, 0.99 for
  SageAttention 3's and FP4 P·V's, 0.97 for their ratio (predicted > 0.8: **held**). The set where FP4 both beats
  SageAttention 3 is 11 and 10 blocks, 9 in both; the blocks that differ sit at a ratio of 0.99–1.01.
- So the fast mode has two roads: FP4 both with q̄ per frame everywhere (if its correction is cheap in the kernel), or
  a fixed map mixing FP4 both and FP4 P·V (stable enough to fix in advance).

## M4.21 — Q smoothing in attention_fp4 (SMOOTH_Q=512)

**What it attacks:** the error of FP4 Q·Kᵀ, not speed. Q goes into the MMA as q − q̄_B (B: a block of 512 queries,
8 kernel blocks), and q̄_B · (k − k̄) comes back as the **start value of the Q·Kᵀ accumulators** (q̄_B · k̄ is one number
per row: softmax cancels it, and the correction stays small). On the tensor cores the start value is free; the cost
is getting 256 bytes of correction per tile into the right registers.

**Decision:** prototype first, the correction computed by its own pass (`smooth_bias`), then fold it into K's
quantization. The own pass reads K like the quantizers do, with RoPE on the way in, so `fuse_rope` keeps working (the
plain torch GEMM was out: it needs the rotated K in float, 0.7 GB).

**Pieces:**
- `rope_stats(x, rope=None, per=512)`: works without a table; per-block mean, min and max from its 256-token partials.
- `quantize_fp4_rows(..., mean_tokens)`: one mean per block of tokens; the per-head scale from the per-block min/max.
- `smooth_bias`: [n_blocks × 128] · [128 × keys] per head, float32 [heads][n_blocks][padded_keys] (365 MB at
  34 020 × 40 heads). Tensor cores: q̄ and k − k̄ each as two bf16 (hi + lo), q̄·x ≈ hi·hi + hi·lo + lo·hi, error
  ~2^-17 of Σ|q̄ᵢ kᵢ| (~10⁻⁴ of a logit: nothing next to FP4's).
- `attention_fp4` with `UNICA_SMOOTH_Q`: the correction travels by bulk copy (`BIAS_CHUNK` tiles per copy) and two
  LDS.64 per 8 keys fill the accumulator quad. Without smoothing the kernel is the same SASS (2152 instructions; only
  parameter offsets moved).
- Tests: the kernel is the emulator's `smooth_q=512` within 0.1 % (with and without RoPE, partial blocks of queries and
  of copies); on Q with a drifting mean 13.0 % against 29.8 % without smoothing.

**Hypothesis (before measuring):** +35–40 instructions per tile (SASS: 8 LDS + 27 MOV) on ~485 → kernel **+4…+8 %**;
`smooth_bias` ~0.5–1 ms; whole call **+3…+5 ms**, still below SageAttention 3.

**First measurement** (34 020, one process per shape: the first try ran at 14.7 of 16 GB VRAM and WDDM paged, SA3 at
96 ms): kernel +4.3 ms (+8.7 %), quantization +3.7 ms, whole call +8.4 ms: **missed**, SA3 −1.7 % only.

**Where it went, step by step** (each A/B in one process, alternating rounds):

| step | result | hypothesis |
|---|---|---|
| `smooth_bias` on the FP32 pipe, means from L1 | 3.8 ms; ncu: 71 % of stalls on L1 | |
| + `#pragma unroll 8` | 3.0 ms | |
| + means in shared memory | 2.2 ms; shared memory 80 % busy (LDS.128 of 32 key rows: 4 wavefronts each) | |
| keys in registers, read from global | **9.5 ms**: 16 rows per LDG, uncoalesced | wrong, reverted |
| **tensor cores, hi/lo bf16** | 1.65 ms | memory floor ~0.9 ms |
| + means pre-arranged as A fragments (one LDG.128) | **1.59 ms**: occupancy 16.7 % (34.8 KB shared per block) | |
| kernel: one LDS.64 + 2 MOV → two LDS.64 | +4.9 → +4.3 ms | −1…−2 ms: missed |
| diagnosis: copy without using it / using it without copy | +2.4 ms / −0.45 ms | |
| L2 prefetch of the correction, 4 or 16 tiles ahead | +5.9…6.1 ms (worse) | DRAM latency: **refuted** |
| STAGES=3 | +6.6 ms (worse) | |
| issue the copy from warp 1, not warp 0 | +4.2 ms (same) | warp 0's issue path: refuted |
| **one copy per 8 / 32 tiles** (2 buffers, 2n · 256 bytes) | **+3.2 / +2.3 ms** | ≤ +2 ms at 8: missed |

- **A bulk copy has a cost of its own**, whatever its size and wherever its data: 256 bytes per tile cost ~2 ms, an
  L2 prefetch (one more async operation) made it worse. Fewer, larger copies are the fix.
- The "free" load without a copy read shared memory that was never written: likely zeros, the same computation as
  without smoothing. With real values (the first chunk only, no further copies) it costs +2.3 ms. ncu (4 heads,
  32 760): cycles +3.7 %, instructions +7.7 %, LSU instructions +44 %, **no more MIO throttle** (63 → 57 K), more
  `wait` stalls: the 16 LDS sit in front of the first OMMA of each block, on the chain. Next lever: load the correction
  one iteration earlier into registers (+16 registers on 225).
- `smooth_bias`'s remaining 0.7 ms above the memory floor goes away when it is folded into K's quantization (step 2).

**Result, whole call** (one process per shape, 6 alternating rounds; `BIAS_CHUNK=32` now the default):

| | 32 760 | 34 020 | 34 020, fused RoPE |
|---|---|---|---|
| SageAttention 3 | 59.14 ms | 62.68 ms | |
| `attention_fp4` | 49.31 ms (−16.6 %) | 53.31 ms (−15.0 %) | 53.19 ms |
| **+ Q smoothing per 512** | **52.57 ms (−11.1 %)** | **57.28 ms (−8.6 %)** | **57.36 ms (−8.5 %)** |
| of which kernel / quantization | +2.10 / +1.09 ms | +2.14 / +1.32 ms | +4.18 ms in all |

The whole-call hypothesis (+3…+5 ms) **held after the fixes**, the kernel's share (+4.3 % against +4…+8 %) too; the
first version missed both. Error on real WAN: the node now measures the kernels themselves (`fp4`, `fp4_smooth512`)
next to the emulator, `build/wan_attention_errors_m4.21.csv`; expected 7.49 / 7.85 % against SageAttention 3's
8.67 / 8.89 % (M4.20b).

**Result on real WAN** (ComfyUI app's agent, the M4.20b setup: seed 42, `measure_every` 7, the same 22 (step, block)
pairs, the prompt changed by one word against the cache; the kernels themselves, not the emulator):

| | median rel L2 | below SageAttention 3 |
|---|---|---|
| baseline 1 (INT8) | 0.899 % | |
| SageAttention 2 | 1.151 % | |
| SageAttention 3 | 8.646 % | |
| `attention_fp4` | 8.919 % | 12 of 22 |
| **`attention_fp4`, SMOOTH_Q=512** | **7.452 %** | **22 of 22** |
| emulator `smooth_q=512` | 7.463 % | |

- Predicted 7.49 %: **held** (7.45). The kernel is the emulator within 0.025 points in every block.
- Smoothing takes 10 % off FP4 both's error (median ratio 0.90), and puts it below SageAttention 3 in every block:
  median 0.90 of SA3's error, worst block 0.97, best 0.41.
- **By the fast mode's rule this is the first win:** 8.6 % faster than SageAttention 3 at the real 34 020 tokens, 10 %
  less error on real data, in every measured block. It becomes the fast mode's baseline 2 (below).

**Baseline 2, fast mode:** `attention_fp4` with `SMOOTH_Q=512` (NVFP4 Q·Kᵀ and P·V, Q smoothed per 512 queries,
`BIAS_CHUNK=32`, bulk copies), with or without `fuse_rope`. 57.3 ms against SageAttention 3's 62.7 ms at 34 020
(−8.6 %), real WAN 7.45 % against 8.65 %. The accurate mode's baseline stays baseline 1.

## M4.22 — beyond WAN: four more models in ComfyUI

**The question:** does the kernel carry over to other models? It needs head size 128, self-attention without a mask,
and long sequences (it wins from ~4K tokens). Installed and fitting: LTX-2.3 22B (video + audio), Qwen-Image-Edit 2511,
Z-Image Turbo, Moss (all DiTs with heads of 128, `comfy/model_detection.py`). The native node goes through
`optimized_attention_override`, which every ComfyUI model uses, so no model-specific code.

**The node, on the way** (each found by a run, `build/attention_errors_models.csv`, column `tag` per model):
- It logs why it passes a call on (first time per reason, with the shapes) and counts the routes per run (`routes`).
- **Run 1: no model reached the kernel**, all "positional arguments": these models pass `heads` (and `mask`, ...) by
  position, WAN by name. The override now reads ComfyUI's signature (`heads, mask, attn_precision, skip_reshape,
  skip_output_reshape`) either way.
- **Run 2: Z-Image and Moss crashed** in the measurement count: Lumina's context refiners call attention with sigmas but
  no block index, which reset the step to None. Such calls are no longer counted.
- `build/dumps.flag`: while it exists, every measured call also saves its q, k, v, to take an error apart offline.

**Hypothesis:** the accurate mode below SageAttention 2 and the fast mode (Q smoothed per 512) below SageAttention 3 in
most blocks, as on WAN; the smoothing gains less on images with text and image tokens joined.

**Result** (medians of rel L2 against FP32; the kernels themselves, which match the emulator to 0.001 points):

| model | calls to the kernel | tokens | accurate / SA2 | below SA2 | fast / SA3 | below SA3 |
|---|---|---|---|---|---|---|
| LTX-2.3 | 520 of 3115 (cross-attention and the audio part's heads of 64 go on) | 3 520, 14 080 | **0.50 / 0.64 %** | 20 / 20 | **3.72 / 4.24 %** | 18 / 20 |
| Qwen-Image-Edit | 240 of 240 | 8 393 | **1.50 / 1.81 %** | 19 / 20 | **11.05 / 12.41 %** | 20 / 20 |
| Moss | 2640 of 2640 | 3 850, 3 874 | **1.31 / 2.08 %** | 40 / 40 | **11.70 / 13.24 %** | 40 / 40 |
| **Z-Image** | 1626 of 1626 | 4 128 | **9.12 / 2.33 %** | **12 / 68** | 13.65 / 15.36 % | 62 / 68 |

(Z-Image: two runs of 34 measured calls, the second with the dumps.)

- Three models of four behave like WAN: the accurate mode 16–38 % below SageAttention 2, the fast mode 10–20 % below
  SageAttention 3. The smoothing per 512 works on images too (hypothesis on joined text and image: refuted, it gains as
  much). Without it FP4 both would beat SageAttention 3 in 8 / 20, 3 / 20 and 18 / 40 calls only.
- LTX's two fast-mode losses (3.55 against 3.25 %, 3.49 against 3.23 %) sit in calls with outlier channels in Q
  (channel spread 10.4 and 6.7 against the usual 2–3).
- **Z-Image breaks the accurate mode**: 9.1 % median, up to 36.8 %, 4.75× SageAttention 2.

**Z-Image taken apart** (the dumps, emulator, one part exact at a time; call 168, the worst):

| variant | error |
|---|---|
| today (kernel = emulator) | 36.77 % |
| exact Q·Kᵀ / exact P | 36.74 / 36.58 % |
| **exact V** | **4.08 %** |
| V in FP8 | 4.21 % |

**V has outlier tokens**: a token's largest |v| is 65–249× the median token's in the bad blocks (2–4× in the good
ones). INT8 V has one scale per channel over all keys, set by the outliers, and the other tokens get a few of its 127
steps. SageAttention 2's FP8 V has relative precision and does not care. The FP4 path (V block-scaled per 16 keys in
e4m3) holds too: the fast mode is still below SageAttention 3 there.

**Options, emulated on Z-Image's 34 dumped calls** (SageAttention 2: median 2.33 %):

| V | median | max | below SA2 |
|---|---|---|---|
| INT8, scale per channel (today) | 9.12 % | 36.77 % | 6 / 34 |
| INT8, scale per channel and 1024 / 256 / 64 keys | 7.02 / 6.35 / 6.22 % | 21.3 / 19.8 / 19.4 % | 6 / 34 |
| P and V in FP8 (SageAttention 2's formats) | 1.59 % | 3.63 % | 33 / 34 |
| exact V, FP16 P·V (`attention_int8qk`) | 0.98 % | 2.74 % | 34 / 34 |
| **INT8 per token, the scale folded into P (P' = p · s_j), P' u8 per tile of 64** | **0.44 %** | **0.76 %** | **34 / 34** |

(The last row is a plain model with an exact Q·Kᵀ, so it misses INT8 Q·Kᵀ's share, ~0.2–0.3 points in the good blocks.)

- Scales per block of keys do not help: the outliers are everywhere, not a few tokens.
- **A scale per token, folded into P**, does: V8_j = v_j / s_j and P'_ij = p_ij · s_j give the same P·V, and P' is
  already quantized per tile against its own max. Folding s_j into P is adding log2 s_j to key j's score before the
  exponential: per key, the same for every query, which is what M4.21's accumulator start does. The open part is the
  softmax denominator: Σ p_j is no longer Σ P̂'_j. To be designed.

**The idea: a low v_offset itself says that V's centring and scaling do not suit these blocks** (Spearman between
`v_offset` and unica / SA2: Z-Image −0.86, Qwen −0.43, Moss +0.35, LTX +0.30, WAN +0.15). Tested against the outlier
explanation on the 34 dumps, with one plain model for every variant (Q·Kᵀ exact, P u8 per tile of 64 as the kernel,
only V changes; today's accurate mode does not centre V at all):

| V | median | max | below SA2 | ρ(v_offset, error / SA2) |
|---|---|---|---|---|
| a: int8 per channel (today) | 9.04 % | 36.8 % | 6 / 34 | −0.86 |
| b: minus the channel mean, added back | 9.92 % | 42.3 % | 6 / 34 | −0.90 |
| c: minus the mean per channel and 64 keys | 10.84 % | 44.0 % | 6 / 34 | −0.91 |
| e: clamped to the 99.9th percentile | 8.29 % | 37.3 % | 6 / 34 | −0.40 |
| **d: int8 per token, scale in P** | **0.44 %** | **0.76 %** | **34 / 34** | +0.75 |
| d after b's centring | 0.36 % | 0.67 % | 34 / 34 | +0.85 |

- v_offset and the outlier ratio go together almost perfectly (ρ = −0.96): v_offset is the channel means' RMS over
  V's RMS, and outlier tokens inflate the denominator. **Held fixed against each other, v_offset keeps nothing**
  (partial ρ +0.10), **the outliers keep it** (partial ρ +0.60).
- Centring makes it worse (b, c), clamping barely helps (e: the outliers are signal, attended to); a scale per token
  fixes it, and what is left grows with v_offset instead (+0.75: the good blocks keep their ~0.7 %).
- Synthetic, on the best call (block 5, 0.88 %): the channel means removed (v_offset 0.34 → 0.00) leave 0.89 %; 1 % of
  the tokens scaled ×100 around the mean (v_offset falls to 0.04 by itself) give 5.98 %, and d keeps 0.71 % on both.
- The other models' lowest v_offset calls (0.17–0.23) sit at unica / SA2 0.56–0.94: a low v_offset alone does no harm.
- So the idea pointed at the right blocks: a low v_offset is the symptom (cheap to compute, a usable detector), the
  outlier tokens are the cause, and centring is not the cure.

## M4.23 — V with one scale per token in the accurate mode (V_TOKEN=1)

**Decision:** replace the per-channel V scale, with a way back (`TILE["V_TOKEN"]`, the native node's `v_scale`).

**What it attacks:** the error from V's outlier tokens (M4.22), not speed. V8_j = round(v_j / s_j), s_j = max |v_j| / 127
over the token's channels; s_j goes into P as log2 s_j added to key j's score, so P' = p · s_j is quantized against
its own tile max and every V row keeps its 127 steps. The denominator Σ p = Σ P' / s_j cannot stay on IDP4A (8-bit
weights 1 / s_j would span the outliers' ~250×): it is summed in float before rounding.

**Emulator first** (`emulate(v_token=...)`, today's accurate kernel otherwise):

| | Z-Image, median | max | below SA2 | random 4 × 4096 |
|---|---|---|---|---|
| per channel (today) | 9.12 % | 36.8 % | 6 / 34 | 1.697 % |
| per token, Σ P̂' / s_j | 1.21 % | 2.83 % | 33 / 34 | 1.604 % |
| **per token, Σ P' / s_j in float** | **1.10 %** | **2.80 %** | **34 / 34** | **1.604 %** |

**Pieces:** `quantize_v(per_token=True)` (the 16 lanes of a token find its max by xor shuffles; per pair of keys a
float4 {log2 s, log2 s, 1 / s, 1 / s}, zeros past the keys; the per-channel scale becomes ones and its `aminmax` pass
goes); `online_softmax_u8_token` (x = s · scale + log2 s_j, the tile max over x, P' = 255 · 2^(x − max), the
denominator FFMA'd with 1 / s_j); the 8 float4 per tile read straight from L2 (every block of a head reads the same, so
no copy: M4.21's lesson). `tests/test_v_token.py`: the kernel is the emulator to the third decimal, random and with
outlier tokens, and in the caller's layout. On Z-Image's 34 dumped calls, the kernel itself: **median 1.10 %, max
2.80 %, below SageAttention 2 in 34 of 34** (per channel 9.12 / 36.8 %, 6 of 34).

**Hypothesis (before measuring):** +~56 instructions per tile (a FFMA per element for log2 s_j, one for the
denominator, IDP4A gone) and 8 LDG.128: kernel **+8…+12 %**; quantization **−~0.3 ms** (no `aminmax`); whole call at
34 020 **+6…+9 ms**.

**Machine code:** the loop 642 → 721 instructions per tile (+79: FADD +32, FFMA +32, LDG +8, IDP4A −8, the rest
addressing); **registers 235 → 255 with 12 / 16 bytes of spills**.

**Result** (one shape per process, 6 alternating rounds):

| | 32 760 | 34 020 | hypothesis |
|---|---|---|---|
| SageAttention 2 | 67.66 ms | 74.06 ms | |
| per channel | 71.60 ms (+5.8 % vs SA2) | 76.97 ms (+3.9 %) | |
| **per token** | **79.79 ms (+17.9 %)** | **86.19 ms (+16.4 %)** | |
| kernel | +7.78 ms (+11.1 %) | +7.85 ms (+10.3 %) | +8…+12 %: **held** |
| quantization | −0.36 ms | −0.39 ms | −0.3 ms: **held** |
| whole call | +8.19 ms | +9.21 ms | +6…+9 ms: at the top, 34 020 a little over |

- Better everywhere on error (random 1.70 → 1.60 %, Z-Image 9.1 → 1.1 %), **11 % slower in the kernel**: by the accurate
  mode's rule (speed against SageAttention 2 at the same error) not a win, so baseline 1 stays per channel and
  `TILE["V_TOKEN"]` stays 0; the native node runs per token by default, as decided, and per channel with `v_scale`.
- Where the time goes, for later: the FADD per element (x + offset before ex2), the denominator's FFMA, the 255
  registers. A cheaper variant would keep the per-channel path and switch per call or per layer on a detector
  (v_offset or the outlier ratio, M4.22).

**On the five models in ComfyUI** (ComfyUI app's agent; `build/attention_errors_v_scale.csv` has both accurate
kernels in every row; medians and the worst call):

| model | per channel | **per token** | SageAttention 2 | per token below SA2 | below per channel |
|---|---|---|---|---|---|
| Z-Image | 7.95 % (35.86) | **1.05 % (2.99)** | 2.07 % (4.15) | 34 / 34 | 34 / 34 |
| WAN 2.2 | 0.92 % (1.63) | **0.78 % (1.57)** | 1.51 % (2.33) | 22 / 22 | 22 / 22 |
| Moss | 1.32 % (1.87) | **1.10 % (1.60)** | 2.08 % (2.56) | 40 / 40 | 32 / 40 |
| Qwen-Image-Edit | 1.50 % (2.28) | **1.15 % (2.01)** | 1.81 % (2.63) | 20 / 20 | 20 / 20 |
| LTX-2.3 | 0.50 % (1.42) | **0.47 % (1.36)** | 0.63 % (1.69) | 20 / 20 | 17 / 20 |

A whole render (measure_every 0, one run each, warm model, ±1 s): WAN 47.9 → 49.7 s (+3.8 %), LTX 38.9 → 40.5 s
(+4.1 %), Z-Image 11.2 → 11.1 s, Moss 10.0 → 10.1 s, Qwen 4.7 → 4.7 s.

- Per token is below SageAttention 2 in **every** measured call of all five models (136 of 136), at 0.59–0.80 of its
  error, and below the per-channel kernel in 125 of 136: the outliers are Z-Image's extreme, but every model has some.
- The kernel's +11 % is +4 % of a whole render on the video models and within the noise on the image ones, whose
  sequences (4–8K tokens) leave attention a smaller share.
- **Another bug of the measurement, found by this run:** M4.22's fix for Lumina skipped every call without a block
  index, and LTX numbers none: no LTX row at all. Calls without one are now skipped only in a model that numbers blocks
  elsewhere (`Measurement.numbered`), and LTX is counted call by call again; tested.

## M4.24 — "Adaptive V Scale Grouping", and where V's outliers sit

**The idea:** the per-channel error comes mostly from a few tokens with extreme V, so neighbouring tokens could share a
scale in small groups: most of per token's precision with fewer scaling operations.

**Cost first, from the kernel's structure:** M4.23's +11 % are per element of P (log2 s_j added to a score, the
denominator weighed by 1 / s_j), and a lane goes through each of its elements whatever the group size: a group below a
tile (64 keys) saves nothing. A group of 64 is free (the scale folds into the tile's beta, IDP4A stays).

**Accuracy against g** (emulator, `v_group`, Z-Image's 34 dumped calls; SageAttention 2 2.33 %):

| g | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| median | 1.10 % | 2.05 % | 2.18 % | 4.01 % | 6.03 % | 10.16 % | 10.27 % |
| below SA2 | 34 | 16 | 16 | 10 | 8 | 6 | 6 |

Grouping breaks at g = 2 already, and the free g = 64 is as bad as per channel: the idea in this form is **refuted**.

**Why, and what the idea pointed at:** the outliers are not ~1 % of the tokens. In Z-Image they are **two tokens,
positions 1 and 2 of 4128**, the top of every head in 28 of 34 calls (the sequence starts with the caption: likely
attention sinks); their neighbours are text that the queries look at, so sharing their scale costs much. The
adaptive form of the idea: take the top-T tokens of a head by max |v| out of INT8 (the channel scales from the other
tokens, their V8 rows zero) and add their part of P·V exactly, once, after the loop, with the final row max. The main
loop stays today's per-channel one.

| (emulator) | Z-Image | WAN | Moss | Qwen-Image-Edit | LTX-2.3 |
|---|---|---|---|---|---|
| a token's max / median \|v\| | 65–249 | 2.2–5.9 | 3.3–6.8 | 2.1–7.2 | 2.1–5.4 |
| per channel | 9.12 % | 0.91 % | 1.32 % | 1.50 % | 0.50 % |
| top 2 out | **1.14 %** | 0.90 % | 1.25 % | 1.50 % | 0.48 % |
| top 8 out | **1.12 %** | 0.90 % | 1.21 % | 1.46 % | 0.46 % |
| top 32 out | | 0.90 % | 1.19 % | 1.30 % | 0.43 % |
| per token (M4.23) | 1.10 % | 0.76 % | 1.10 % | 1.15 % | 0.48 % |
| SageAttention 2 | 2.33 % | 1.51 % | 2.08 % | 1.81 % | 0.63 % |

(other models: the dumps of M4.23's runs, 22 / 40 / 20 / 20 calls; their top tokens move from call to call.)

- **Two mechanisms.** Z-Image has a couple of extreme tokens: taking 2–8 out gives per token's accuracy for ~nothing in
  the loop. The other models have no extreme tokens (2–7×), and per token's 15–23 % there come from the spread of all
  tokens' sizes: taking tokens out barely moves them (WAN 0.91 → 0.90 %).
- So the choice is between: per channel + top-T out (Z-Image fixed, the rest as today, below SageAttention 2 already;
  cost: a tail after the loop, ~1 tile in ~530, and a top-T in the quantization), and per token (every model 15–23 %
  better, +11 % kernel, +4 % of a video render).

## M4.25 — V's largest keys out of INT8 (V_OUTLIERS=T)

**Decision:** both mechanisms ("we have two modes anyway"): the native node's `v_scale` gets `outliers` next to
`per_token` (still the default) and `per_channel` (the way back).

**Pieces:**
- `emulate(v_outliers=T)`, as the kernel does it: the T keys of a head with the largest max |v| are zeroed in V before
  its channel scales are taken; in the loop their P̂ goes into the denominator like any key's; after the loop their part
  of P·V is added, 255 · 2^(x − m) with x their INT8 score and m the final row max, times their V row over the scale.
  Z-Image's dumps: 1.17 % median for T = 2 … 8 (34 / 34 below SageAttention 2).
- `unica.v_outliers`: the keys by a per-token max (`aminmax` over the channels, no copy of V) and the channel scale of
  the others, exact: `block_stats` (rope_stats' partials, per 256 keys) for every block, and the ≤ T blocks that hold an
  outlier taken again from their own rows. `quantize_v(skip=...)` writes the outliers' rows as zeros.
- `add_v_outliers`, after the loop: lane l scores row l % 16 against keys l / 16, + 2, ... with IDP4A on Q8 in shared
  memory and K8 in global memory (the loop's score exactly), and adds P · v into its 32 columns, each row's P by shuffles.
- `tests/test_v_token.py`: the scale bit for bit the emulator's; the kernel the emulator's to the third decimal (two
  "sink" tokens ×200: 12.9 → 1.34 %).

**Hypothesis (before measuring):** the loop not changed by one instruction, the tail ~1000 instructions once per warp
against ~340 000 in the loop: kernel **+0.2…0.5 %**; one more pass over V and a few small ops: quantization
**+0.4…0.6 ms**; whole call at 34 020 **+0.5…1 ms** (per token: +9 ms).

**Machine code:** the loop 1284 instructions per two tiles either way (99 lines differ by register names); registers
235 → 238, no spills; the tail +904 static instructions.

**Result** (one shape per process, 6 alternating rounds):

| | 32 760 | 34 020 | hypothesis |
|---|---|---|---|
| kernel | +0.14 ms (+0.2 %) | −1.10 ms (−1.4 %, noise) | +0.2…0.5 %: **held** |
| quantization | +0.99 ms | +1.01 ms | +0.4…0.6 ms: **missed** (2×: the torch steps, topk, the blocks' gather, scatter, are ~5 more launches) |
| whole call | +1.52 ms (+2.1 %) | +2.39 ms (+3.1 %) | +0.5…1 ms: missed, with the quantization |
| per token, for comparison | +8.08 ms | +9.47 ms | |

The kernel itself on Z-Image's 34 dumped calls: per channel 9.12 % (max 36.8), **outliers T = 2 / 8: 1.17 % (max
3.23 / 2.98)**, per token 1.10 % (2.80); all three of the latter below SageAttention 2 in 34 of 34.

- `outliers` costs ~1 ms per call, a fifth of per token's ~9 ms, and fixes the model with extreme tokens; per token stays
  the more accurate one on every model. The quantization's extra 1 ms can come down by folding the per-token max into
  block_stats' pass and the block recount into one small kernel.

**On the five models in ComfyUI** (ComfyUI app's agent; `build/attention_errors_v_modes.csv`, the three accurate modes
in every row; medians (max), and the calls below SageAttention 2):

| model | per channel | outliers | per token | SageAttention 2 | below SA2: outliers / per token |
|---|---|---|---|---|---|
| Z-Image | 7.95 % (35.86) | 1.11 % (3.23) | **1.05 % (2.99)** | 2.07 % | 34 / 34, 34 / 34 |
| WAN 2.2 | 0.92 % (1.63) | 0.92 % (1.62) | **0.78 % (1.57)** | 1.51 % | 20 / 22, 22 / 22 |
| Moss | 1.32 % (1.87) | 1.23 % (1.65) | **1.10 % (1.60)** | 2.08 % | 40 / 40, 40 / 40 |
| LTX-2.3 | 0.50 % (1.20) | 0.49 % (1.20) | **0.48 % (1.15)** | 0.63 % | 18 / 20, 20 / 20 |
| Qwen-Image-Edit | 1.50 % (2.28) | 1.46 % (2.27) | **1.15 % (2.01)** | 1.81 % | 19 / 20, 20 / 20 |

A whole render, two rounds, warm model (s): per channel / outliers / per token

| | per channel | outliers | per token |
|---|---|---|---|
| WAN | 48.3 / 48.3 | 48.4 / 48.2 (+0.0) | 49.7 / 49.6 (+1.4, +2.9 %) |
| Z-Image | 11.0 / 10.9 | 11.5 / 11.5 (+0.5, +5 %) | 11.0 / 11.0 (+0.0) |
| Moss | 10.0 / 10.1 | 10.7 / 10.7 (+0.7, +7 %) | 10.1 / 10.1 (+0.0) |
| Qwen-Image-Edit | 4.7 / 4.7 | 4.7 / 4.8 | 4.7 / 4.7 |
| LTX-2.3 | 35.9 / 45.4 | 35.8 / 37.0 | 41.4 / 37.5 (±5 s of noise: no conclusion) |

- **The costs run the other way round on images.** `outliers` adds a fixed ~0.3 ms per call (~12 small launches of its
  quantization: top-k, the blocks' gather, the masks), which the image models' ~1 700–2 500 short calls (~4K tokens)
  multiply into +5–7 %, while per token's +11 % of a short kernel disappears there. On WAN's long calls it is the other
  way: `outliers` free, per token +2.9 %.
- **Per token is the more accurate everywhere** and the only one below SageAttention 2 in every call (136 / 136,
  `outliers` 131 / 136). `outliers` matches it only on Z-Image, an image model, where per token costs nothing.
- So, on these five models, **per token is the default to keep**; `outliers` earns its place only on a long-sequence
  model with extreme tokens (none among these), and only after its quantization is one or two kernels.

## M4.26 — a reproducible bench on the five models

Decision: no more work on `outliers` for now. First a release of what already works on the five models, with numbers
anyone can reproduce. Until now the runs went through workflows that exist only on this machine. Now
[bench/comfyui_models.py](bench/comfyui_models.py) runs five self-contained graphs from
[bench/comfyui/](bench/comfyui/). The graphs fix seeds, prompts, sizes and steps; their models and where they come
from are in [models.md](bench/comfyui/models.md).

What the run does, per model:

- free ComfyUI's memory;
- one measured run, which gives the errors of the three V scales on the same calls;
- three rounds of seven modes, in alternating order:
  - `off`: ComfyUI's own attention, SDPA without flags;
  - `sage2`, `sage3`: KJNodes' patch sends every attention call to SageAttention 2 (`auto`) or 3;
  - `per_channel`, `per_token`, `outliers`: the accurate mode;
  - `fast`: attention_fp4 with Q smoothed per 512. It is now a `mode` input of the native node, next to `v_scale`.

The time is the median of the three renders, because one LTX render varies by ±5 s. Every CSV row carries the
environment: the kernel's git version, ComfyUI, torch, CUDA, the driver, the GPU and its power limit.

What was taken out of the studio workflows:

- the SageAttention 3 patch under the node on WAN, so calls the node passes on go to SDPA, not to SA3;
- Gemma's rewriting of the LTX prompt (time, and a text that could change between runs);
- the switched-off RIFE and upscaler.

**Trial run** (one round, so the times are single renders):

- The errors match M4.25. Per token is below SageAttention 2 in 135 of 136 calls; the one other is a WAN call, 21/22,
  with SA3 no longer under the node.
- The image models are 16–18 % faster than `off` on Z-Image and Qwen. On Moss, the same as SDPA (9.9 vs 10.1 s): there
  Unica wins on error against SageAttention 2, not on time.
- WAN: 49.3 s vs 65.6 s.
- LTX: per_channel +23 % and outliers +35 % against `off`, per token +1.6 %; M4.25 had no such gap. It may be ComfyUI
  loading weights between renders rather than the kernel; three rounds will tell.
- ComfyUI crashed once, in its dynamic VRAM (aimdo.dll, 0xc0000005), between LTX's second sampling stage and the
  decode. It was the first such crash in 14 days, and the same graph passed again. The bench now waits for the
  restart, loads the models again untimed and goes on, instead of stopping an hour-long run.

**Full run on e0bc502** (median of 3 renders in seconds; error: median rel L2, and in how many measured calls it is
below the rival: SageAttention 2 for the accurate modes, 3 for fast). WAN comes from a second run of its own: during
the first one, round 0 shared the GPU with a check of the kernels.

| model | off | sage2 | sage3 | per_channel | per_token | outliers | fast |
|---|---|---|---|---|---|---|---|
| WAN | 68.2 | 45.6 | **44.2** | 48.6 | 50.2 | 48.3 | 44.5 |
| LTX | 43.7 | 37.9 | 40.3 | 40.6 | 41.4 | 45.1 | 43.0 |
| Qwen | 5.7 | 4.8 | **4.6** | 4.8 | 4.8 | 4.9 | 4.7 |
| Z-Image | 12.9 | **10.8** | 11.1 | 11.0 | 11.1 | 11.7 | 11.4 |
| Moss | 9.9 | **9.7** | 10.1 | 10.1 | 10.2 | 10.8 | 10.5 |

| error | sage2 | per_token | sage3 | fast |
|---|---|---|---|---|
| WAN | 1.51 % | **0.78 %**, 21/22 | 8.60 % | **8.16 %**, 22/22 |
| LTX | 0.58 % | **0.42 %**, 20/20 | 3.44 % | **3.10 %**, 17/20 |
| Qwen | 1.81 % | **1.14 %**, 20/20 | 12.37 % | **10.94 %**, 20/20 |
| Z-Image | 2.07 % | **1.05 %**, 34/34 | 15.29 % | **12.13 %**, 32/34 |
| Moss | 2.08 % | **1.10 %**, 40/40 | 13.18 % | **11.73 %**, 40/40 |

- **On error, Unica wins everywhere:** per token is below SageAttention 2 in 135 of 136 calls, fast below
  SageAttention 3 in 131 of 136.
- **On the time of a whole render, it does not.** On WAN, fast ties SageAttention 3 (44.5 vs 44.2 s), but the accurate
  modes are 6–10 % slower than SageAttention 2 (per token 50.2 vs 45.6 s). This is not noise: the rounds agree within
  ±1 s. Elsewhere it is level within a few tenths of a second. This contradicts the kernel benchmarks, and the cause is
  not known yet. Candidates, to measure, not to assume:
  - the quantization and torch launches around the kernel;
  - cross-attention: the KJNodes patch sends it to SageAttention too, while Unica passes it to SDPA;
  - SageAttention being cheaper on the short calls.
- **Fast is the slower mode on short sequences.** `python -m unica` on [1, 16, 4096, 128]: fast 1.3–1.7 ms, per token
  0.63 ms. On WAN's long calls it is the faster one. This is why it loses on Moss and Z-Image.
- **Runs drift by 3–8 % between them.** The second WAN run was slower than the first in every mode, the untouched ones
  too (off 66.2 → 68.2 s). Modes compare only within one run. The bench now writes the SM clock and the temperature
  after each render.
- **LTX's medians are not reliable.** Within one mode, renders spread from 35 to 53 s, and the first render after
  loading took 81 s. LTX 22B and Gemma do not fit in 16 GB together, so weights move in and out between renders. Its
  errors are clean.
- **Moss is level everywhere, SageAttention included:** its calls are short, and attention is not what takes the time.

## M4.27 — the package: one wheel for Windows and Linux, a compatibility check

The idea: anyone with the card installs it with pip, nothing to build, and is told plainly when it will not run.

- **One wheel, `py3-none-any`.** A cubin is the GPU's own machine code, the same file for the driver on Windows and on
  Linux. The wheel ships the four cubins the node runs (`unica.PRESETS`):
  - plain `flash.cubin`: per channel, fp4 and all the quantizers;
  - `V_TOKEN=1`, `V_OUTLIERS=8`, `SMOOTH_Q=512`.

  [build_wheel.py](build_wheel.py) compiles them fresh from `csrc/` and prints their sha256, so that WSL can check it
  loads the very same bytes. An installed package has no `csrc/` and takes the shipped cubins. A setting outside them
  says so, instead of looking for nvcc.
- **The driver library loads on first use.** Until now `import unica` loaded `nvcuda.dll` straight away, so on a
  machine without the driver the import failed, and the ComfyUI node with it.
- **`unica.compatibility()`** has three answers:
  - **cannot run:** compute capability other than 12.0 (sm_120a code is for 12.0 only), a driver for CUDA 12, no
    driver, torch without CUDA;
  - **untested:** another 12.0 card, a driver older than the tested one, torch other than 2.14, Linux (until WSL);
  - **ok:** otherwise.

  The package warns once on import. The nodes then:
  - leave the model's own attention in place if it cannot run;
  - log a warning if it is untested;
  - send every later call to the model's own attention if a call to unica raises.
- **`python -m unica`** prints the verdict and runs each preset against FP32 on random inputs. The thresholds catch a
  broken kernel, not a worse one: on N(0, 1) the fast mode loses 19.26 %, because random data is FP4's worst case
  (M4.2: P·V alone 13.7 %).
- **Checked from a clean environment outside the repository, without nvcc on PATH:**
  - with a CPU-only torch it says "cannot run" and fails nothing;
  - with torch 2.14+cu132 the four presets give errors bit for bit the same as in the repository (1.58 / 1.67 / 1.66 /
    19.26 %);
  - an unshipped setting gives the error above.

## M4.28 — why WAN's renders were slower than SageAttention: a sync in the node

M4.26 left the accurate modes 6–10 % behind SageAttention 2 on WAN, and fast level with SageAttention 3. The kernel
benchmarks said otherwise.

**Step 1: one WAN call by every path** ([bench/wan_call.py](bench/wan_call.py)). q, k, v are
[1, 34020, 40 · 128] fp16, as ComfyUI's WAN passes them; cross-attention to 512 text tokens. A render has 160 calls of
each kind. Hypotheses, written before measuring:

| part | hypothesis | measured |
|---|---|---|
| fp16 → bf16 of q, k, v and back (WAN's activations are fp16, M4.17 assumed bf16) | 3.5 ms | 3.58 ms alone, ~2.5 ms inside the path |
| per token's kernel against SageAttention 2 | +9 ms (M4.23: +16 %) | +13 ms (88 vs 75 ms) |
| per channel against SageAttention 2 | 0 | +4.6 ms (79.6 vs 75 ms) |
| cross-attention: SDPA against SageAttention 2 | +1 ms | +1.6 ms (3.5 vs 1.95 ms) |

Also measured: fast against SageAttention 3 on self-attention is −5.8 ms (55.0 vs 60.8 ms). On cross-attention,
SageAttention 3 is the slowest of the three (5.8 ms).

Per render, this predicted per channel +1.0 s, per token +2.4 s and fast −1.3 s. The renders had +3.0, +4.6 and
+0.3 s. That leaves ~2 s in every Unica mode that one call alone does not show.

**Step 2: ComfyUI's log, step by step.** Per sampling step (40 blocks), per channel took 10.05 s against SageAttention
2's 8.96 s. The calls predicted +0.25 s. So in the running model, each Unica call cost 10–25 ms more than alone.

**Step 3: the cause.** `Measurement.due()` ran on every self-attention call, with `measure_every` 0 too. It read
`float(sigmas.max())`, and `sigmas` is on the GPU, so the read is a device-to-host copy. The CPU waits there until the
GPU has done everything queued. The GPU then idles while Python queues the rest of the block: norms, linears, RoPE.
One call alone cannot show it, since nothing is queued after it. The off and sage modes do not use the node, so they
never paid it. Fix: off, `due()` returns at once and reads nothing; a test checks that sigma is not touched.

This is host-side code. Neither the kernel nor its SASS changes, and Nsight Compute has nothing to show here: the cost
is the GPU waiting between kernels, not inside one.

**Step 4: WAN again, 3 rounds** (sage2, sage3, per_channel, per_token, fast), with hypotheses written before:

| mode | render | step (high / low noise) | hypothesis | verdict |
|---|---|---|---|---|
| sage2 | 45.9 s | 9.34 / 9.37 s | — | — |
| sage3 | 46.6 s | 9.31 / 9.23 s | — | — |
| per_channel | 47.9 s | 9.94 / 9.84 s | within ~1 s of sage2 | no: +2.0 s, but half the gap is gone (+0.6 s per step, was +1.09) |
| per_token | 49.4 s | 10.08 / 10.07 s | ~+2.4 s against sage2 | close: +3.5 s |
| fast | **44.6 s** | **9.05 / 9.11 s** | ~−1.3 s against sage3 | yes, more: −2.0 s |

The errors are the same as in M4.26.

- **Fast is now the fastest mode on WAN**, ahead of both SageAttention 2 and 3, and below SageAttention 3's error in
  22 of 22 calls.
- **The accurate modes are still behind SageAttention 2.** Per channel: +0.6 s per step where the calls predict +0.25.
  Per token: +0.7 against +0.58. What is left:
  - the fp16 conversion (~0.1 s per step);
  - per token's kernel;
  - cross-attention through SDPA;
  - ~0.35 s per step for per channel that is still unexplained.
- **Runs drift within a run, too.** Every mode got faster from round 0 to round 2, by 0.3–0.5 s per step; SageAttention
  2 went from 9.74 to 9.22. That is the size of the differences sought. The temperature stayed at 42–44 °C, so it is not
  heat. The SM clock the bench wrote was read after the render, at idle, so it shows nothing. The bench now samples
  the clock every 2 s while a render runs and writes their median.

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

## M4.29 — the wheel on Linux: WSL2 on the same card

The decision: before PyPI and any rented card, the wheel has to work on Linux, since nearly every cloud runs it. Checked
in WSL2 (Ubuntu 22.04, kernel 6.6) on this RTX 5080: the Windows driver 617.14 serves Linux through
`/usr/lib/wsl/lib/libcuda.so.1`. Stage A only: uv, Python 3.12, torch 2.14.0+cu132 and the wheel from
`build_wheel.py`. No CUDA toolkit, no nvcc, which is a user's machine.

- **The same bytes.** The four shipped cubins have the sha256 that `build_wheel.py` printed on Windows.
- **The same results.** `python -m unica` gives 1.58 / 1.67 / 1.66 / 19.26 %, bit for bit the Windows errors. Its one
  remark was "untested: platform linux".
- **The same speed.** Hypothesis before measuring: within ~2 % of Windows, the GPU and the driver being the same. The
  kernel on WAN's shape [1, 40, 34020, 128] bf16, the same script, Windows and WSL alternating twice (median of
  6 rounds × 5 runs):

| preset | Windows | WSL |
|---|---|---|
| per_token | 91.4 / 93.5 ms | 91.5 / 92.2 ms |
| per_channel | 81.9 / 82.8 ms | 82.2 / 82.9 ms |
| outliers | 83.4 / 84.0 ms | 83.6 / 84.4 ms |
| fast | 56.5 / 57.1 ms | 57.9 / 58.3 ms |
| one small call (host cost) | 0.17 ms | 0.12–0.13 ms |

  It held: within 1–2 %, and the host cost of a call is lower on Linux.
- **PyTorch's SDPA is not the same.** On `python -m unica`'s shape it took 1.46 ms in WSL against 3.5 ms on Windows: the
  Linux build has the flash backend. On a Linux machine, "off" in the ComfyUI bench is a stronger baseline than here.

`compatibility()` now counts Linux as tested. What WSL does not check is a native Linux driver: the first rented card
will.

## M4.30 — the quality test on "wan stable work": blur from the workflow, and a VRAM overflow that reset the driver

The idea: compare the attention modes on a real quality workflow, not the bench's fast one. That workflow is "wan
stable work" (WanVideoWrapper, three passes):
- high noise at CFG 3.5 over sigmas 1 → 0.9;
- a finishing pass with Lightning;
- a hires pass at 1 MP from sigma 0.9;
- Nomos ×2 and RIFE ×5 at the end.

The run used the old prompt and negative, seed 42, and no LoRA on the high model: with it, loading crashes, filling RAM
and VRAM. Each mode got one render.

**Blur everywhere, the reference included.** Past the first frame, every mode's video is soft, with hatched hair and
melting edges, SageAttention 2's (the workflow's own attention) as much as the others. So the blur comes from the
workflow without its high-noise LoRA, not from the attention. The likely cause is 4 steps of the full high-noise model,
too few to build sharp motion.

**Relative to SageAttention 2**, on the hires pass before upscaling and interpolation (113 frames, 768 × 1360):

| mode | PSNR mean (worst frame) | flicker | by eye |
|---|---|---|---|
| sageattn_3 | 24.80 dB (21.33) | 5.79 | the face melts, the hand doubles |
| unica accurate | **28.62 dB (26.31)** | **4.16** | closest to the reference, the same blur |

The fast render was stopped before its hires pass ended.

**The hires pass is 118,320 tokens per call, and it overflowed the VRAM:**

| render | hires pass | reserved VRAM peak |
|---|---|---|
| sageattn | 84 s per step | 13.5 GB |
| sageattn_3 | 167–182 s per step | 14.4 GB |
| unica accurate | ~245 s per step | 15.6 of 16 GB in use, 23 GB of shared memory |

The GPU showed 99 % utilization at 100–180 W, far below what real compute draws under the 360 W limit. Windows does not
fail on an overflow: it moves part of the memory to system RAM, read over PCIe.

At 118 k tokens, one copy of q, k or v is 1.21 GB. Unica's path holds about 7.9 GB besides the model's own tensors:
- q, k, v converted to bf16: 3.63 GB;
- Q8, K8, V8: 1.8 GB;
- the output in bf16, then in fp16: 2.4 GB.

SageAttention 2 holds about 3 GB. To be measured: the peaks of both paths on this shape.

**During the fast render the driver reset the GPU context** (System log, nvlddmkm ID 153, 04:08:34). The screen showed
broken glyphs, then the taskbar kept a stale dark colour, and only a reboot cleared it. There was no other driver error
in three days, apart from 03.11: a misaligned address, our own kernel bug during development, fixed then. The memory
runs at its stock 15,001 MHz. Hypothesis: the overflow made single kernel launches take seconds over PCIe, and Windows
reset the context.

What follows:
- no Unica hires runs until the memory is fixed;
- reading fp16 directly in the quantizers and writing fp16 out of the kernel goes from an optimization to a necessity;
- huge calls get split by heads, so that one launch stays well under a second;
- the LoRA comes back with `merge_loras = false` in WanVideoWrapper: applied on the fly, with no merge at load.

## M4.31 — a call split by heads to stay in the VRAM, with a reserve for the desktop

The decision: before anything else, no mode may overflow the VRAM again, at any length (M4.30: a reset driver and a
broken desktop until reboot).

**Step 1: the peaks.** [bench/wan_call.py](bench/wan_call.py) `--memory 118320`: one self-attention call of the hires
pass, fp16, memory above q, k, v (1.13 GiB each). Hypotheses written before:

| path | hypothesis | measured | time |
|---|---|---|---|
| SDPA | — | 1.13 GiB | 2.80 s |
| SageAttention 2 | ~3 GB | 3.95 GiB | 1.00 s |
| SageAttention 3 | ~5 GB | **27.9 GiB** | 2.81 s |
| Unica per channel | ~7.9 GB | 6.21 GiB | 1.19 s |
| Unica per token | ~7.9 GB | 6.24 GiB | 1.15 s |
| Unica fast | ~7.0 GB | 9.75 GiB | **38.7 s** |

- **Fast overflowed** on an empty 16 GB card: 3.4 GiB of inputs, 9.75 GiB of its own and the desktop's 2.5 GB. One call
  took 38.7 s over PCIe. That is the render that reset the driver (M4.30).
- **The hypothesis missed fast's q̄ · k table (M4.21):** float32 per head, per block of 512 queries, per key. It grows as
  the square of the length: 0.36 GB at 34,020 tokens, 4.4 GB at 118,320.
- **SageAttention 3 needs 27.9 GiB** for this call, so on a 16 GB card it overflows too. That is why it was twice as
  slow as SageAttention 2 on the hires pass.

**Step 2: split by heads.** Every head is computed on its own (its scales, its outliers), so a call can run a few heads
at a time. `unica.attention` now:
- sizes the pieces from the free VRAM: the driver's free memory, plus what torch holds unused, minus a reserve;
- takes fp16 and converts each piece to bf16 on its own, instead of the node copying all of q, k, v;
- writes one output in the caller's dtype and layout.

Before the change, a check that the free-memory query does not wait for the GPU, after M4.28. With ~204 ms of matmuls
queued, `torch.cuda.mem_get_info` returned in 0.32 ms of host time and left the 204 ms queued. A real sync, `float()` of
a GPU tensor, waited the whole 205 ms.

Tests ([tests/test_pieces.py](tests/test_pieces.py)): for every preset, pieces of 3, 3 and 2 heads, and fp16 converted
piece by piece, are bit for bit one bf16 call. The output keeps ComfyUI's [batch, tokens, heads, 128] layout.

**Step 3: what splitting costs** ([bench/pieces_cost.py](bench/pieces_cost.py), WAN's 34,020 tokens, forced pieces).
Hypothesis: within 3 % down to 8 heads per call.

| preset | 20 heads | 10 heads | 8 heads | 5 heads |
|---|---|---|---|---|
| per token | −0.1 % | 0.0 % | −1.0 % | +0.5 % |
| per channel | −0.3 % | −1.2 % | −0.9 % | +0.1 % |
| outliers | −0.2 % | −1.1 % | −1.1 % | −0.1 % |
| fast | +1.1 % | +3.3 % | +3.4 % | +4.6 % |

It held for the accurate modes, which stay within noise. Fast pays slightly more: smooth_bias and Q's statistics have a
fixed cost per piece.

**Step 4: the reserve.** The first reserve was 512 MB. In a fresh process, fast was sized to 35 heads: the driver
reported 10.12 GiB free, which leaves a 9.6 GiB budget, and the peak was 9.66 GiB. The estimate per head was right
(296 MB measured, 291 estimated), but the call used the VRAM to its last megabyte. After SageAttention 3 had run in the
same process, it overflowed again (27 s). On Windows, the desktop needs more room than that.

Since small pieces cost next to nothing, the reserve is now the larger of 1 GiB and a tenth of the card: 1.6 GiB on
16 GB. `UNICA_VRAM_RESERVE_MB` or the nodes' `vram_reserve_mb` (0 means automatic) set it by hand. The same call at
118,320 tokens, right after SageAttention 3:

| path | peak above the inputs | time |
|---|---|---|
| SageAttention 2 | 3.95 GiB | 0.91 s |
| Unica per channel | 7.18 GiB | 0.98 s |
| Unica per token | 7.22 GiB | 1.08 s |
| Unica fast | 6.98 GiB | **0.64 s** (38.7 s before) |

Next: fp16 read directly by the quantizers and written by the kernel, which removes the bf16 copies and the second
output. The q̄ · k table for fast is computed inside the kernel instead (decision pending).

## M4.32 — the free VRAM counted without torch's unused pool: the hires pass of "wan stable work" at SA2's speed

The idea: rerun "wan stable work" with the M4.31 split, now with the high-noise Lightning LoRA back
(`wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise`, Comfy-Org's repack, applied with `merge_loras = false`, strength 1;
torch.compile off, inductor does not build with MSVC). Same prompt, negative and seed 42 as M4.30. Without merging,
loading no longer fills the RAM and the VRAM.

**The split did not hold in the real workflow.** SageAttention 2 went through: high 88 s, Lightning 30 s, hires
(118,320 tokens) 263 s, about 66 s per step (84 in M4.30). Unica accurate was stopped by the guard on the hires pass:
529 s and still running, with shared GPU memory +1.7 GiB in its first 20 s.

Two hypotheses, written before measuring:
- H1: the free VRAM is overestimated. free_vram counted what the driver has free plus what torch holds unused. Under
  WDDM that pool need not be in the VRAM: with SageAttention 2, torch held 15.4 GiB while 11.8 GiB of the card was in
  use. Prediction: 20+ heads per call on the hires pass, and a 6 GiB reserve brings it under 320 s.
- H2: the system RAM is the limit (63.4 of 63.5 GiB with SageAttention 2), and the extra shared memory goes to the
  pagefile. Prediction: page reads jump with Unica, and a larger reserve barely helps.

A log of every sizing decision (`UNICA_LOG_VRAM=1`) on the hires pass:

| run | reserve | heads per call | hires pass | shared growth |
|---|---|---|---|---|
| A, automatic | 1.6 GiB | 35 mostly (also 5, 37, 26, 14, 3) | stopped at 529 s | +1.7 GiB |
| B | 6 GiB | 1 always | 260 s | 0 |

- **H1 held.** At the end of the hires pass the driver had 0.00 GiB free, and the 35 heads came from torch's 7.5 GiB
  of unused pool alone. The card was full (15.25 of 16 GiB dedicated), so part of that pool sat in system RAM. A call of
  35 heads touches ~3 GiB of buffers, and WDDM evicted something else (model blocks) for them, on every call.
- **H2 refuted.** Page reads were as high in B (93 k/s) as in A (129 k/s), and B ran at SageAttention 2's speed. That
  paging is block swap's, not ours.
- B was 1.0× SageAttention 2, better than the predicted 1.2×: one head at 118 k tokens is 85 MB of work, and the split
  is next to free for the accurate mode (M4.31).

**The fix:** free_vram is now what the driver reports free, and nothing else. When the card is full, a call gets one
or a few heads.

| mode | high | Lightning | hires | shared growth | heads per call at 118 k |
|---|---|---|---|---|---|
| SageAttention 2 | 88 s | 30 s | 263 s | +0.07 | — |
| Unica accurate (per token) | 95 s | 31 s | **259 s** | 0 | 1 (6519×), 12 (4×), 4 (1×) |
| Unica fast | 83 s | 29 s | **197 s** | 0 | 1 (6519×), 7 (6×), 5 (1×) |

Predicted: accurate within 276 s (1.05× SageAttention 2), fast within 263 s. Both held, fast with 25 % to spare. No
driver errors in any run.

**Quality against SageAttention 2** (hires before upscaling, 113 frames of 768 × 1360):

| mode | PSNR mean (worst frame) | flicker |
|---|---|---|
| Unica accurate | 25.69 dB (21.89) | 5.07 |
| Unica accurate, reserve 6 GiB | 25.70 dB (21.90) | 5.06 |
| Unica fast | 22.55 dB (20.31) | 6.19 |

- With the LoRA, every mode is sharp: clean eyes, hair and collar, none of M4.30's smearing. M4.30's blur was the
  missing LoRA.
- Accurate matches SageAttention 2 by eye (the same smile and strands on frame 92).
- Fast drifts in the story (mouth closed, another expression and tilt), and fine horizontal ripples show in the hair
  and face that the others do not have. Its real error at 118 k tokens was never measured; M4.21 measured 34 k.

**The two accurate renders are not bit for bit the same** (38.1 dB between them). They differ only in how the calls
were split, and pieces were supposed to be bit for bit one call. A check on one 118 k call, 12 heads, in pieces of 12,
4 and 1:
- fast is bit for bit the same;
- accurate differs by 1.2e-4 at most, 1 ulp.

The cause is torch's `k.mean(-2)`, used by the accurate mode when RoPE is not fused. From ~32 k tokens its summation
order depends on how many heads the tensor holds: 3e-9 apart at 32,760 and 118,320 tokens, equal at 700, the length
tests/test_pieces.py used. A few int8 roundings of K move. Over 40 blocks and ten steps that 1 ulp grows to 38 dB.
Fast takes its statistics from its own kernel (rope_stats), so it does not have this.

Open:
- make accurate deterministic across pieces: K's mean from rope_stats, and the test at 32 k tokens;
- SageAttention 2 rendered twice, to know what PSNR run-to-run noise alone gives;
- fast's error on real QKV at 118 k tokens, and the ripples.

## M4.33 — microbenchmarks on the workflows' call signatures

The idea: no renders for now. Check the open questions of M4.32 with microbenchmarks on the call shapes the workflows
make, write the hypotheses first, then measure. The signatures (self-attention, 40 heads, d = 128, fp16 in ComfyUI's
[batch, tokens, heads, 128]):
- 29,928 tokens: "wan stable work", high-noise and Lightning passes;
- 34,020 tokens: the bench's Rigsy workflow (22 real q, k, v dumps exist, M4.22);
- 118,320 tokens: "wan stable work", hires pass (no dumps: random data).

Hypotheses, written before measuring:

| | what | prediction |
|---|---|---|
| H1 | accurate's K mean from rope_stats (own kernel, fixed order) instead of torch's k.mean | bit for bit across pieces of 1, 4 and 12 heads at all three lengths; the mean pass costs within ±0.2 ms of k.mean at 34 k (both read K once, ~0.35 GB); error on the 22 WAN dumps changes by < 0.005 pp |
| H2 | efficiency over length (whole call as the node runs it, % of today's peak: INT8 460, FP4 935) | fixed costs amortize: at 118 k accurate per token ≥ 57 % and fast ≥ 46 % (55 / 43 % at 34 k); 29,928 within 1 pp of 34,020 |
| H3 | the split at 118 k: pieces of 1 head against 10 | ≤ +1 % for accurate, ≤ +2 % for fast (a few small launches per piece against ~25 ms of work per head) |
| H4 | fast's ripple: is its error structured by the 512-query smoothing blocks or by the token rows (54 tokens per row at 34 k)? | no: on the WAN dumps the error of a block's first and last 16 queries is within 5 % of its middle, and the per-query error ratio fast / accurate has no spectral peak at period 54 or 512 (peak < 3× the median) |

H3 missed (numbers below), so one more, written before measuring:

| | what | prediction |
|---|---|---|
| H3b | the 1-head penalty is the tail of the last wave: a head at 118,320 tokens is 1849 blocks of 64 queries, 11 waves of 168 (84 SMs × 2) plus one block alone | at 118,272 tokens (1848 blocks, exactly 11 waves) pieces of 1 head cost ≤ +1 % against 10 |

All in [bench/signatures.py](bench/signatures.py), one run per part, the GPU otherwise idle.

**H1 held: accurate is now bit for bit the same however a call is split.** The accurate mode takes K's mean from
rope_stats, the same pass fast already used. No kernel changed.

| tokens (12 heads) | k.mean | rope_stats | pieces of 4 and 1 = one call, k.mean / rope_stats |
|---|---|---|---|
| 29,928 | 0.121 ms | 0.143 ms | no / **yes** |
| 34,020 | 0.137 ms | 0.162 ms | no / **yes** |
| 118,320 | 0.426 ms | 0.481 ms | no / **yes** |

- About +0.08 ms per 40-head call at 34 k, within the predicted ±0.2 ms.
- On the 22 WAN dumps the error is 0.8052 % either way (largest change 0.00002 pp).
- tests/test_pieces.py now also splits a 34,020-token call of 4 heads into pieces of 1, for every preset. It fails
  with the old k.mean and passes now.

**H2 held, barely: longer calls are only 1–3 points more efficient.** The whole call as the node makes it, fp16 in
[batch, tokens, heads, 128], random data, 3 alternating rounds:

| path | 29,928 | 34,020 | 118,320 |
|---|---|---|---|
| per channel | 63.7 ms, 62.6 % | 84.2 ms, 61.2 % | 970 ms, 64.3 % |
| per token | 70.5 ms, 56.5 % | 90.4 ms, 57.0 % | 1087 ms, 57.4 % |
| fast | 45.2 ms, 43.4 % | 57.2 ms, 44.3 % | 659 ms, 46.6 % |
| SageAttention 2 | 59.9 ms, 66.5 % | 78.4 ms, 65.7 % | 922 ms, 67.6 % |
| SageAttention 3 | 45.4 ms, 43.2 % | 58.3 ms, 43.5 % | not run (27.9 GiB) |

Percentages are of today's peaks: INT8 460 TOPS, FP4 935 TFLOPS (bench/peak.py).
- Fast ties SageAttention 3 at 30 k and 34 k, and is 1.40× SageAttention 2 at 118 k.
- Accurate per channel is 5–7 % behind SageAttention 2 here, and per token 15–18 %. The hires pass in M4.32 hid this
  (259 vs 263 s) because attention is only part of each step.
- At 29,928 per channel is 1.4 points from 34 k, not within 1.

**H3 missed, H3b held for fast.** Pieces of 1 head against 10, 118,320 tokens:

| tokens | blocks per head | per token | fast |
|---|---|---|---|
| 118,320 | 1849 = 11 × 168 + 1 | +2.7 %, then −2.7 % on a rerun | +5.0 %, +5.4 % |
| 118,272 | 1848 = 11 × 168 | +0.6 % | +2.0 % |

- For fast, about 3 of its 5 points are the last wave: one block running alone after 11 full waves, once per piece.
  The other 2 points are its fixed cost per piece (Q's statistics, smooth_bias).
- For per token the run-to-run spread (±3 %) is larger than the effect, so no conclusion.
- In the M4.32 hires render 6519 of 6524 calls ran as 1-head pieces, so fast paid ~5 % there.

**H4: no row structure, and the smoothing blocks leave only a faint mark.** Per-query error on the 22 WAN dumps at
34,020 tokens:
- Edges against middles of the 512-query blocks: fast 6.575 / 6.445 % (1.020), per token 0.997. Within the predicted 5 %.
- The fast / accurate ratio by position within a block, in eighths: 1.016 1.000 0.996 0.992 0.993 0.994 0.999 1.011.
  A shallow U, ±1.5 %, against ±0.3 % on a control period of 504. The ends of a block lie farther from its mean q̄,
  since neighbouring tokens are alike.
- Spectrum of the fast / accurate ratio (peak / median): period 54 (a row) 1.99, held; period 512 3.52, missed (that
  U); period 1620 (a frame) 12.6, the strongest in 15 of 22 dumps.

Fast's extra error follows the content: the same places of every frame. It does not follow rows or the kernel's
blocks. So the fine row ripple seen in the M4.32 render is not a kernel pattern at 34 k. Hires dumps (118 k) would
settle it at that length.

Next, by these numbers:
- accurate stays behind SageAttention 2 by 5–7 % (per channel) and 15–18 % (per token) per call;
- fast at hires could merge its 1-head pieces into one grid (the wave tail, ~3 %) or pick pieces that fill whole waves.

## M4.34 — real q, k, v from "wan stable work": the hires pass at 118,320 tokens

The idea: H4 of M4.33 (fast's ripple) was checked only on the bench's 34 k dumps. One render of "wan stable work"
(accurate per token, measure_every 13, 2 heads) saved the measured q, k, v. The guard stopped the hires pass after 95 s:
shared memory +1.85 GiB, where M4.32's render without measuring stayed at +0. The likely cause is the measurement
itself (FP32 reference, SageAttention 3 and the emulator on top of a full card). Saved: 30 calls at 29,928 tokens and 3
at 118,320 (calls 403, 416, 429).

Hypothesis, written before measuring: H4 holds at both lengths as at 34 k. The error of a 512-block's edges is within 5 %
of its middle. The fast / accurate ratio has no peak at the row period (43 and 85 tokens; peak < 3× the median), and the
frame (1032 and 4080 tokens) is the strongest period.

**The measurement is what overflowed.** One measured call at 118,320 tokens, 40 heads, 2 of them measured, on an
otherwise empty card: 2.47 GiB above the inputs, 31.7 s. On a full card during a render that is enough to spill to
shared memory. Without measuring, M4.32's render stayed at +0. A hires run with every measurement would need a lighter
measurement there, for example without SageAttention 3 and the emulator.

**Errors on the real hires calls** (the node's CSV, rel L2 against FP32, median):

| | 29,928 tokens (30 calls) | 118,320 tokens (3 calls) |
|---|---|---|
| Unica accurate per token | **0.694 %** | **0.834 %** |
| Unica accurate per channel | 0.779 % | 0.953 % |
| SageAttention 2 | 0.958 % | 1.233 % |
| Unica fast (fp4_smooth512) | **6.744 %** | **7.461 %** |
| SageAttention 3 | 7.095 % | 8.039 % |

Per call at 118 k, Unica / SageAttention 2: 1.205 / 2.110, 0.834 / 1.233, 0.655 / 0.603 %. That last call is the only
one in this render where SageAttention 2 is closer.

**H4 held at both lengths** ([bench/signatures.py](bench/signatures.py) `ripple stable`, `ripple hires`):

| | 29,928 | 118,320 |
|---|---|---|
| fast, 512-block edges / middles | 1.020 | 1.014 |
| row period (43 / 85 tokens), peak / median, median (max) | 2.05 (3.03) | 2.26 (3.20) |
| frame period (1032 / 4080 tokens) | 9.59 | 12.48 |
| by position within 512 queries (eighths) | ±1.3 % | ±0.7 % (control 504: ±0.4 %) |

The period-512 peaks (4.70 and 5.46) are the frame's harmonics at these lengths, not the blocks:
- at 29,928, 1032 / 2 = 516 falls in the same frequency bin as 512;
- at 118,320, 4080 / 8 = 510 is within one bin of it.

The position profile, which harmonics do not touch, is flat to ±0.7 % at hires.

Fast's per-query error is not larger at hires (6.1 % against 6.4 % at 34 k), and like there it follows the content of
each frame. So the fine row ripple of M4.32's fast render does not come from any position pattern of the kernel's
error. It is the size of FP4's error, carried through 10 denoising steps, as the drift of fast's render in the story
(mouth, tilt) shows. Fast stays below SageAttention 3 at every length measured.

## M4.35 — the tail of "wan stable work": Nomos ×2 and RIFE ×5, microbenchmarks

The idea: about 250 s of the ~640 s render lie outside the three sampling passes (M4.32), more than any gain left in
attention (accurate against SageAttention 2 on the hires pass is ~15 s). The tail:
1. VAE decode of the hires pass: 113 frames of 768 × 1360.
2. Nomos ×2 (`2xNomosUni_span_multijpg_ldl`, SPAN: a CNN with no softmax attention, so nothing for our kernel). ComfyUI
   runs it in fp32, in tiles of 512 with an overlap of 32.
3. RIFE ×5 (`rife49`, ComfyUI-Frame-Interpolation, `fast_mode` on, `ensemble` off) on 1536 × 2720 frames: 112 pairs,
   561 frames out.
4. VideoCombine at 60 fps.

The RIFE node's loop (`vfi_utils._generic_frame_loop`), per pair and per middle frame:
- it uploads both frames again in fp32, 50 MB each: 8 uploads per pair instead of 2;
- `.detach().cpu()`, a sync, after every middle frame;
- `gc.collect()` after every pair;
- its output is float32 on the CPU: 561 × 50 MB ≈ 28 GB of RAM.

A quirk: it calls `model(f0, f1, t, scale_list, fast_mode, ensemble)`. The fifth positional parameter of
`IFNet.forward` is `training`, so `fast_mode` lands there and `fastmode` gets `ensemble`. A faster loop must call it the
same way to give the same frames.

Hypotheses, written before measuring (the hires render's first frames; the per-pair times scale to 112 pairs):

| | what | prediction |
|---|---|---|
| H1 | Nomos as ComfyUI runs it | ≤ 0.3 s per frame, ≤ 35 s for 113 frames (≤ 15 % of the 250 s) |
| H2 | RIFE as the node runs it | the largest part: ≥ 1.3 s per pair, ≥ 150 s for 112 pairs (≥ 60 % of the 250 s) |
| H3 | the same fp32 model, each frame uploaded once, the 4 middle frames of a pair in one batch, copied back to pinned memory without a sync, no gc.collect | ≥ 1.5× faster than the node, frames equal to it (largest difference ≤ 1e-5) |
| H4 | H3 with fp16 autocast | ≥ 3× faster than the node, PSNR ≥ 40 dB against its frames |

First run ([bench/rife_loop.py](bench/rife_loop.py), 13 frames of the hires render, 12 pairs, extrapolated to the
render's 113 frames):

| | per frame / pair | for the render | hypothesis |
|---|---|---|---|
| Nomos, as the node (fp32, tiles of 512) | 0.127 s | 14.3 s | ≤ 35 s: held |
| Nomos, whole frames, fp32 | 0.093 s | 10.5 s | PSNR 80.2 dB against the node |
| Nomos, whole frames, fp16 | 0.053 s | 6.0 s | PSNR 70.5 dB |
| RIFE ×5, as the node | 0.362 s | **40.5 s** | ≥ 150 s: **missed, by 3.7×** |
| RIFE, batched, fp32 | 0.255 s | 28.6 s | 1.42×, ≥ 1.5×: missed; largest difference 5.3e-3 (PSNR 95.4 dB), not ≤ 1e-5 |
| RIFE, batched, fp16 autocast | 0.219 s | 24.5 s | 1.65×, ≥ 3×: missed; PSNR 56.0 dB, held, but one pixel off by 0.34 |

- RIFE is not where the 250 s go: with Nomos it is ~55 s, and a better loop would save ~12–16 s of it.
- The batched fp32 frames differ from the node's in the last bits. Convolutions run in TF32 by default, and cuDNN picks
  other algorithms for a batch of 4 than for 1.
- One gc.collect() takes 105 ms even in this small process, so ~12 s of the node's 40 s. In ComfyUI's larger heap it
  is probably longer.

H5, written before measuring: VideoCombine encodes 561 frames of 1536 × 2720 at 60 fps with libx264 (VHS h264-mp4: its
default preset, CRF 19, yuv420p, a bt709 scale filter) on the CPU, and converts each float32 frame to bytes in Python
first. Prediction: ≤ 10 frames per second, so ≥ 55 s for the render.

| | per frame | for the render | hypothesis |
|---|---|---|---|
| VideoCombine: float32 → bytes (numpy, one thread) | 0.037 s | 20.8 s | |
| VideoCombine: bytes and x264 together, as VHS does | 0.046 s (21.6 fps) | **26.0 s** for 561 frames | ≥ 55 s: **missed** |

x264 runs alongside the Python conversion and mostly waits for it: 37 of the 46 ms per frame are numpy.

**The tail measured so far is ~80 s, not 250:** Nomos 15 s, RIFE 39 s, VideoCombine 26 s. A second run gave RIFE
0.344 s per pair (38.6 s), so it varies by ~5 %. The other ~170 s are in what these benchmarks do not cover:
- loading the two 14B models with LoRA and block swap;
- the text encoder;
- four VAE decodes (965 and 972, the high and Lightning previews, still decoded with save_output off; 981, the hires
  pass; 1064);
- gc.collect in ComfyUI's larger heap.
Fixing RIFE's loop would save ~10–15 s of a 640 s render. A render timed per node has to come first.

**ComfyUI's environment.** ComfyUI always runs with `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
`NUMEXPR_NUM_THREADS` set to 8. The CPU is a Ryzen 9 9950X3D: 16 cores, 32 threads, and the 3D V-cache sits on one of
its two 8-core chiplets. torch takes 8 threads with these variables and 16 without. The runs above had none of them.
Rerun with all four (the second run, so the ~5 % spread applies): Nomos 14.0 s, RIFE 38.7 s, batched fp32 1.36×,
batched fp16 1.53×, VideoCombine 24.9 s. The tail does not change: it waits on the GPU and on single-threaded numpy.
Every CPU benchmark from now on runs with these variables.

**Where the other ~170 s may go.** Theories, each with a number, to be checked:

| | theory | prediction |
|---|---|---|
| T1 | the text encoder: umt5-xxl, fp16, on the CPU (`CLIPLoader device: cpu`), 8 threads. Zen 5 has no fp16 arithmetic, so torch takes a slow path | ≥ 20 s per render for both prompts; 16 threads ≤ 1.6× faster (V-cache on one chiplet); bf16 ≥ 2× faster than fp16. ComfyUI caches the node while the prompt is unchanged, so only a render after a restart or with a new prompt pays it |
| T2 | loading the two 14B fp8 models from disk after every `/free`, and pinning them for block swap (40 blocks, non_blocking) | 30–60 s per render: 2 × ~14 GB from the disk, plus a pinned copy |
| T3 | VAE: WanVideoEncode (1065) of all 113 frames at 1 MP before the hires pass, and the hires decode (981), both untiled, bf16 | 40–80 s together |
| T4 | the decodes 965 and 972: the high and Lightning passes, decoded for preview videos with save_output off | 10–20 s for frames nobody keeps |
| T5 | gc.collect in ComfyUI's heap: the RIFE node calls it 112 times | ≥ 0.2 s per call in ComfyUI against 0.1 s here, so ≥ 20 s |

**Plan for the next session:**
1. Microbenchmark of the text encoder on the CPU (T1), no GPU needed: 8 and 16 threads, fp16 and bf16, the render's
   prompt and negative.
2. One render timed per node by the ComfyUI agent (websocket `executing` events), launched as ComfyUI always is, with
   the four variables. It checks T2–T5 at once and shows whether the text encoder ran or was cached.
3. Fix what the timings rank first. Likely candidates, cheapest first:
   - drop the decodes 965 and 972, or keep them only when the previews are wanted (T4);
   - the text encoder in bf16, or on the GPU before the models load (T1);
   - RIFE's loop: one upload per frame, batched middle frames, no gc.collect per pair. 10–15 s, frames equal to
     PSNR 95 dB in fp32;
   - VideoCombine's float32 → bytes on the GPU (37 of its 46 ms per frame are numpy).
4. Then the full cycle with the changes, against the same render without them, in the same session.

## M4.36 — the text encoder of "wan stable work" (T1), microbenchmark

**Evidence before measuring.** ComfyUI's log of the 2026-11-07 render: `Requested to load WanTEModel` at 01:16:52,
`load device: cpu ... dtype: torch.float16`; the next node (WanVideoEncode, 9 s by its progress bar) ends at 01:17:36.
So the text encoder took ~33 s of that render, on the CPU, for the prompt and the negative.

**Hypothesis (T1, refined).** The cost is umt5-xxl's matmuls in fp16 on a CPU without fp16 arithmetic:
- encoding both prompts at 8 threads, fp16: 25–35 s (the log's ~33 s minus the load from the disk);
- bf16 at 8 threads: ≥ 2× faster than fp16 (Zen 5 has AVX-512 BF16, torch's fp16 CPU path converts);
- 16 threads: ≤ 1.6× faster than 8;
- on the GPU, fp16: < 1 s once the weights are there; the cost is the upload of 10.8 GB at ~25 GB/s, ~0.5 s pinned,
  a few seconds pageable.

**What ComfyUI actually computes on the CPU** (probe of `torch.nn.functional.linear`): all 168 Linear calls run in
fp32 × fp32. `comfy.ops.Linear` keeps the weights in the file's dtype and casts each weight to fp32 on every call,
since `should_use_fp16` is false for the CPU. The `dtype` model option does not reach the model: `wan.te(dtype_t5)`
overrides it with the dtype detected in the file. Every prompt is padded to 512 tokens (`min_length=512`), so the two
prompts are ~8.8 TFLOP of matmuls whatever their length.

Where the 2026-11-07 render got `device: cpu`: the runner's API graph (`wan_stable_work_raw_api.json`) sets it on
CLIPLoader. The workflow saved in the UI has `default`, which with aimdo means the GPU.

**Result** (`bench/text_encoder.py`, ComfyUI's .venv-13, the four thread variables at 8; both prompts, 2 runs each):

| path | time, s | rel L2 vs the render's path |
|---|---|---|
| CPU, fp16 weights (the render), 8 threads | 28.06, 27.98 | 0 |
| CPU, fp16 weights, 16 threads | 18.68, 18.51 | 0 |
| CPU, fp32 weights (no per-call cast), 8 threads | 26.05, 25.25 | 0 |
| CPU, fp32 weights, 16 threads | 16.43, 15.73 | 0 |
| GPU, fp16: first call (upload + setup), then | 8.94, 0.67 | 0.001 % |

Against the hypothesis:
- 25–35 s at 8 threads: held (28.0 s). 8.8 TFLOP in 28 s is 0.31 TFLOPS, ~12 % of the fp32 peak of 8 Zen 5 cores.
- "bf16 ≥ 2× faster than fp16": the premise was wrong. There is no fp16 arithmetic to replace: ComfyUI computes in
  fp32 on the CPU. The per-call cast costs 2.5–2.8 s (~10 %). bf16 arithmetic on the CPU would need a patch to
  ComfyUI, so it was not measured.
- 16 threads ≤ 1.6×: held (1.51×, 1.56× without the cast).
- GPU < 1 s once loaded: held (0.67 s). The first call took 8.9 s, worse than "a few seconds" for the upload.

So the text encoder costs ~28 s of a render whenever the prompt changes or ComfyUI restarted, and ~9 s on the GPU.

**Next: three renders timed per node** (by the ComfyUI agent, fresh prompt each, `/free` before each): A = CLIPLoader
`cpu`, B = `default` (GPU), A2 = `cpu` again, to separate the disk cache of the second render from B's gain.
Prediction, written before the renders:
- the CLIPTextEncode nodes: A and A2 28–33 s together, B 9–12 s, so B's render is 17–24 s shorter than A2's;
- the encoder on the GPU must leave the VRAM before the 14B models load; if aimdo does not evict it cleanly, B shows
  "loaded partially" or a larger shared-memory peak in the high pass, and that cost can eat the gain;
- the per-node table checks T2–T5 with the numbers already written above (model loads 30–60 s, VAE 1065 + 981
  40–80 s, preview decodes 965 + 972 10–20 s, RIFE's gc.collect ≥ 20 s inside the RIFE node's ~39 s + overhead).

**The three renders** (ComfyUI agent, 2026-11-08, unica accurate, seed 42, prompt suffixes r1/r2/r3, `/free` before
each; node time = its `executing` event to the next; no cached nodes; CSVs in build/m436_nodes_{A,B,A2}.csv):

| node | | A (cpu) | B (GPU) | A2 (cpu) |
|---|---|---|---|---|
| 976 | WanVideoSampler, hires | 288.4 | 277.1 | 274.4 |
| 1051 | RIFE VFI ×5 | 100.9 | 106.4 | 117.7 |
| 934 | WanVideoSampler, high | 97.9 | 94.8 | 95.0 |
| 981 | WanVideoDecode, hires | 80.8 | 88.6 | 83.6 |
| 984 | VHS_VideoCombine | 49.2 | 34.1 | 47.6 |
| 969 | WanVideoSampler, Lightning | 32.3 | 31.4 | 33.7 |
| 948 + 947 + 946 | text encoder: load, prompt, negative | 36.0 | 3.9 | 38.4 |
| 1049 | Nomos ×2 | 20.6 | 19.2 | 18.4 |
| 977 | I2V encode (hires) | 15.9 | 16.1 | 15.0 |
| 1065 | WanVideoEncode (before hires) | 14.3 | **47.4** | 15.6 |
| 965 + 972 | preview decodes | 11.4 | 12.0 | 12.2 |
| 1064, 937, others | | 13.1 | 15.7 | 12.6 |
| | total | 764.6 | 749.3 | 767.4 |

Peaks: VRAM 10.5 / 13.5 / 10.5 GiB, torch reserved 14.0 / 14.9 / 13.9, ComfyUI RSS 42.4 / 31.3 / 30.6 GiB.
Run-to-run noise, from A against A2: up to ±17 s on one node (RIFE 101 vs 118, hires 288 vs 274), 3 s on the total.

Against the prediction:
- the text encoder on the CPU, 28–33 s: held (33.1 and 31.9 s for the two encode nodes; 36.0 / 38.4 with the
  load). The disk cache does not help it (A2 ≈ A): it is compute.
- on the GPU, 9–12 s: missed, in the good direction, 3.9 s. aimdo does not upload the 10.8 GB up front ("prepared
  for dynamic VRAM loading. 10835MB Staged"); the weights stream in as the layers run.
- B 17–24 s shorter than A2: the total is 18 s shorter, but not for the predicted reason. The encoder saves 34.5 s
  and WanVideoEncode 1065 loses 31.8 s (29 frames at 1.86 s/it against 2.1–2.2 it/s in A and A2); the remaining
  −15 s (VideoCombine −13.5, RIFE −11) is within the noise. With one B render, the net gain is ~0 ± 15 s.
- the cost of the encoder on the GPU shows up where it was not predicted: no "loaded partially" line, no extra
  shared memory in the high pass (B and A2 bases 21.2–21.6 GiB, +0.0–0.12), the WAN models load as in A. It lands in
  WanVideoWrapper's own VAE encode, +3 GiB of VRAM peak. Hypothesis (unverified): the staged encoder stays in the
  VRAM, and WanVideoWrapper's VAE allocates outside ComfyUI's model management, so nothing asks aimdo to evict it.
- T2 (model loads 30–60 s): not visible as nodes; the loaders take < 0.6 s and block swap loads inside the samplers.
  Not checked.
- T3 (1065 + 981, 40–80 s): held above the range, 95 s in A. The hires decode 981 alone is 81–89 s.
- T4 (965 + 972, 10–20 s): held, 11–12 s.
- T5 (RIFE's overhead in ComfyUI ≥ 20 s): held, much larger. The node takes 101–118 s against 39 s in the
  microbenchmark of the same loop (M4.35): 62–79 s go somewhere inside ComfyUI.

Outside sampling the render spends ~345 s, not ~250 (M4.35 counted from the render's total). Ranked: RIFE 101–118,
hires decode 81–89, VideoCombine 34–49 (25 s in the microbenchmark), text encoder 36 on the CPU, Nomos 18–21,
VAE encodes 30–47, previews 12.

## M4.37 — where RIFE loses 62–79 s inside ComfyUI

The RIFE node took 101–118 s in the three renders of M4.36; the same loop took 39 s in M4.35's microbenchmark, which
ran 13 frames and scaled the time per pair to 112 pairs.

**What scales differently at 113 frames.** `_generic_frame_loop` preallocates its output with
`torch.zeros(5 × 113, 3, 1536, 2720)` in float32: 28.3 GB, plus the node's 5.7 GB input (Nomos's frames, float32).
At 13 frames that was 3.3 GB. In all three renders the system RAM peaked at 63.5 of 64 GiB, with ComfyUI's RSS at
30–42 GiB before the node.

**Hypotheses, before measuring:**
- H1 (RAM): the 28.3 GB output does not fit next to ComfyUI's ~30 GB, so Windows pages. In a clean process the node
  runs 113 frames in ≤ 45 s (the 13-frame scaling holds); with ~30 GB of ballast allocated first it takes ≥ 90 s,
  and the page faults and the pagefile grow by tens of GB.
- H2 (gc.collect, T5): the node calls gc.collect after every pair, 112 times; in ComfyUI's heap each call takes
  0.2–0.5 s, so 20–55 s. It does not depend on RAM, so it shows only in the real process.
H1 is checked here first (one process, with and without the ballast); if it does not explain the gap, a sampling
profile of the real render follows.

**H1, measured** (`bench/rife_ram.py`, the node's own loop, bicubic ×2 instead of Nomos, ComfyUI idle in the
background holding its ~30 GB, 26 GiB free at the start):

| frames | output | time | per pair | torch.zeros | gc.collect | free RAM | pagefile |
|---|---|---|---|---|---|---|---|
| 56 | 12.9 GiB | 22.1 s | 0.402 s | 0.6 s | 5.9 s, 55 calls | 21.2 → 8.2 GiB | 3.3 → 3.3 GiB |
| 113 | 26.2 GiB | 53.7 s | 0.480 s | 6.8 s | 12.0 s, 112 calls | 16.2 → 0.0 GiB | 3.3 → 6.4 GiB |

H1 held in kind, missed in size: at 113 frames the free RAM reaches 0, 3.1 GiB go to the pagefile, torch.zeros
costs 11× more for 2× the size, and a pair is 19 % slower. But the node takes 54 s, not ≥ 90 s; the RAM explains
~10 s of the 50–65 s gap. gc.collect is 0.107 s per call in this small process (12 s per node), so H2 needs the real
process.

The remaining ~50 s is inside ComfyUI's process. Main suspect now: the VRAM. aimdo keeps the WAN models staged in
the VRAM after sampling; if RIFE's activations (1536 × 2720 per frame, 4 middle frames per pair) do not fit, WDDM
places them in shared system memory and every pair runs over PCIe. The same mechanism would explain B's slow
WanVideoEncode 1065 in M4.36 (the encoder staged in the VRAM).

**Next: one render profiled per node** (the ComfyUI agent, config B of M4.36, so it also repeats the slow 1065):
py-spy on ComfyUI's process during 1065, 981, 1049, 1051 and 984, plus the GPU sampled once per second (dedicated and
shared memory, utilization, SM clock, power). Prediction, before the render:
- H3 (VRAM): during RIFE the dedicated memory is ≥ 15 GiB and the shared memory grows by ≥ 1 GiB, the GPU
  utilization stays < 60 %, and ≥ 60 % of the main thread's samples sit in the model's forward and in `.cpu()`
  (waiting for the GPU); the same signature in 1065 (B) and not in A's 1065 timing;
- H2: gc.collect takes 20–55 s of RIFE's samples (0.2–0.5 s per call);
- 981 (hires decode, 81–89 s): compute-bound, utilization ≥ 90 %, no shared growth; if instead it shows H3's
  signature, tiling (`enable_vae_tiling`) is the fix to test.

**The profiled render** (ComfyUI agent, config B, suffix r4; 742.3 s; py-spy 0.4.3 `--rate 100 --threads`, without
`--native` (the agent's retry built a wrong command line), no profile of 984; GPU sampled every ~1.3 s; files
build/m437_*):

| node | time | VRAM | util | SM clock | power | WDDM shared |
|---|---|---|---|---|---|---|
| 1065 WanVideoEncode | 14.9 s | 4.1 → 15.9 GiB | 91 % | 2891 MHz | 238 W | flat |
| 981 WanVideoDecode, hires | 69.1 s | 15.6–15.9 GiB | 98 % | 2907 MHz | 133 W | +1.5 GiB over the base |
| 1049 Nomos | 27.5 s | 4.3–6.7 GiB | 49 % | | 119 W | |
| 1051 RIFE | 139.8 s | 4.1–7.2 GiB | 21 % | 2708 MHz | 77 W | flat |
| 984 VideoCombine | 45.6 s | | 5 % | | 33 W | |

RIFE's samples (all inside the node; the line in vfi_utils.py, ~seconds of the 140):

| share | line | what |
|---|---|---|
| 55.6 % | 194 | `gc.collect()` after every pair: ~75 s, ~0.67 s per call |
| 22.0 % | 175 | `.detach().cpu()` of each middle frame: waits for the GPU, then a pageable copy |
| 7.9 % | 172 | `frame1.to(DEVICE)`: both frames uploaded again for each of the 4 middle frames |
| 6.0 % | 147 | `torch.zeros` of the 28.3 GB output under RAM pressure |
| 3.1 % | | the model's forward in Python |

Against the prediction:
- H3 (VRAM) for RIFE: refuted. 4–7 GiB of VRAM, shared flat, GPU at 21 % and 77 W. The node is host-bound.
- H2 (gc.collect 20–55 s, 0.2–0.5 s per call): held in kind, larger: ~75 s, ~0.67 s per call, 6× the 0.107 s of a
  small process. A full collection walks every tracked container in the process, and ComfyUI with its custom nodes
  and caches has many more of them.
- 1065 in B (+32 s in M4.36): not repeated, 14.9 s at 91 % utilization with the encoder on the GPU. M4.36's slow
  1065 was a one-off; the "staged encoder" hypothesis has no support. The encoder on the GPU saves its ~33 s
  (total 742.3 s against A's 764.6 and A2's 767.4).
- 981 "compute-bound, no shared growth": half held. 98 % utilization and Python waits in `images.cpu()` (line 2152,
  the sync after the queued decode), but only 133 W with the VRAM full and +1.5 GiB in shared memory: the GPU is busy
  but starved, part of the decode likely runs over shared memory. 69 s here against 81–89 s in M4.36.

**Decision:** an own node, RIFE VFI (Unica) in comfyui/rife.py: comfyui-frame-interpolation's IFNet, checkpoints and
download, with the loop rebuilt. Each frame uploaded once, a pair's 4 middle frames in one batch, the output
allocated without the zero fill and written in ComfyUI's [n, h, w, 3] layout (the original returns a permuted view of
[n, 3, h, w]), no gc.collect per pair. The model gets the same positional arguments as in the original (fast_mode
lands in IFNet's `training`). tests/test_rife.py checks the frame order against the original's with a stand-in model.

**Hypothesis, before measuring** (it attacks host time: the GPU sat at 21 % in the original):
- standalone (`bench/rife_ram.py`, same process, 113 frames, both loops in turn): the original 54 s, the new loop
  ≤ 35 s: gc.collect −12 s, torch.zeros' fill −6 s, the re-uploads −4 s, the batch ~1.36× on the GPU part (M4.35);
- the frames equal the original's to the rounding of batched convolutions: PSNR ≥ 90 dB, largest difference ≤ 1e-4;
- in the render: the RIFE node 140 s → ≤ 50 s (gc.collect alone was ~75 s there);
- VideoCombine (984) reads a contiguous [n, h, w, 3] instead of the permuted view: ≥ 5 s less of its 45.6 s.

**Standalone result** (`bench/rife_ram.py … 113`, ComfyUI idle in the background, 26.7 GiB free at the start):

| loop | time | per pair | torch.zeros | gc.collect | free RAM min | pagefile max |
|---|---|---|---|---|---|---|
| original | 53.6 s | 0.478 s | 6.1 s | 12.2 s, 112 calls | 0.1 GiB | 6.0 GiB |
| RIFE VFI (Unica) | 28.7 s | 0.256 s | — | — | 7.4 GiB | 7.8 GiB |

- ≤ 35 s: held, 1.87× faster in a small process (where gc.collect is cheap).
- Accuracy on 13 frames: PSNR 98.1 dB against the original, the shapes equal; the largest difference is 3.3e-3,
  above the predicted 1e-4 but below one 8-bit level (3.9e-3). The new output is contiguous, the original's is not.
- This is host code: registers, SASS and ncu do not apply. The GPU side is unchanged except the batch of 4.

**A/B in the render, profiled** (ComfyUI agent, after a restart; config B, both renders with the same prompt
suffix r5, no cached nodes; py-spy `--native` on 1051 and 984):

| | R1, RIFE VFI | R2, RIFE VFI (Unica) |
|---|---|---|
| total | 870.2 s | 765.6 s |
| 1051 RIFE | 249.9 s (util 11 %, 49 W, VRAM 3.8–6.6 GiB) | 142.4 s (util 19 %, 68 W, VRAM 11.0–13.9 GiB) |
| 984 VideoCombine | 104.2 s | 107.2 s |
| 981 hires decode | 26.4 s | 33.8 s |

Not a clean measurement: py-spy with `--native` fell behind ("results may be inaccurate") and inflated the
profiled nodes: the original RIFE took 140 s and VideoCombine 46 s in M4.37 without `--native`. Under the same
overhead the new node is 43 % faster; the predicted ≤ 50 s cannot be judged from this run.
- R2's samples: 34 % in the copy of a batch to the output (it waits for the GPU and takes the page faults of
  the first write into the fresh 28 GB: Windows zeroes pages on demand, so skipping torch.zeros moved that cost here
  rather than removing it), 26 % launching ResConv's kernels from Python, 7 % uploads.
- Frames: the hires videos (before Nomos and RIFE) are bit-identical in R1 and R2. The final videos differ at
  PSNR 40.0 dB, and the input frames, which both nodes copy unchanged, differ just as much (39.9 dB): x264 at crf 19
  reacts to sub-level differences in the neighbouring frames. The node's own output matched at 98.1 dB.
- After the restart the hires decode 981 took 26–34 s against 69–89 s in M4.36–M4.37: its slowness depends on the
  process's state, not on the graph.
- 984 is not faster with the contiguous output (≥ 5 s predicted); judged again in the clean run.

**Next: the same A/B without py-spy**, order reversed (R2 then R1), so that the times are clean.

**What else holds the render back: theories T6–T12**, written while the clean A/B runs. Each has a number, to be
checked. The render with the encoder on the GPU takes ~742–766 s: sampling ~400 s (hires 254–288, high 94–98,
Lightning 31–36), the rest ~345 s.

| | theory | prediction |
|---|---|---|
| T6 | VideoCombine (984, 34–49 s, GPU at 5 %) is numpy turning float32 into bytes one frame at a time on one thread (37 of 46 ms per frame in M4.35), plus page faults reading the 28 GB of frames from a full RAM | conversion in torch with 8 threads, or on the GPU before the copy back: 984 ≤ 15 s, −20…−35 s |
| T7 | the hires pass pages: block swap streams the 14B model's blocks from a RAM that is full (Available 0.8–2.4 GiB, Pages Input/sec up to 44 k in M4.37; 93–129 k/s in M4.32). The spread of hires between renders (254–288 s) follows how full the RAM is (ComfyUI's RSS 29–48 GiB) | hires time grows with the page reads of its own pass; ~10 GiB more free RAM before hires (the encoder off the CPU already gave ~11 GiB) gives ≥ 15 s less |
| T8 | attention is ~60 % of the hires pass (fast 197 s against accurate 259 s at a ~1.6× faster kernel gives ~160 s of attention), and per token's kernel is 15–18 % behind SageAttention 2 per call (57 % against 66–68 % of peak, M4.33) | per token at SageAttention 2's efficiency: −20…−25 s per hires pass; at per channel's (61–64 %): −10…−16 s |
| T9 | the hires decode (981) spills to shared memory when the process is old: 26–34 s right after a restart, 69–89 s later, 133 W at 98 % utilization with +1.5 GiB shared in M4.37. The VRAM is held by models aimdo keeps staged and by torch's cache | freeing the VRAM before 981 (unload the sampler's model, empty the cache) keeps it ≤ 35 s in an old process: −35…−55 s; tiled decode as the fallback, its seams checked |
| T10 | RIFE (Unica) is still host-bound (GPU at 19 % in R2): Python launching ~hundreds of small kernels per forward, and the page faults of writing 28 GB that Windows zeroes on first touch | fp16 autocast 1.65× (M4.35, PSNR 56 dB) or CUDA graphs ≥ 1.5×; the page faults ≥ 5 s of the node, removable only by a smaller output (not ComfyUI's IMAGE contract) |
| T11 | Nomos (1049, 18–27 s) runs fp32 tiles of 512 with an overlap of 32 | whole frames in fp16: 6 s for 113 frames in M4.35 (PSNR 70.5 dB), so −12…−20 s |
| T12 | the first render after a restart pays pinning and loading (RSS 47.8 GiB, shared +3.65 GiB in the high pass, total 870 s in R1 of M4.37b) | the first render ≥ 30 s slower than the next with the same graph; a warm-up render before any timing |

Ranked by the expected gain: T9 (if the process is old), T6, T8, T7, T11, T10, T12. T6, T9 and T11 are workflow
code, T8 is the kernel, T7 is memory.

**The clean A/B** (ComfyUI agent, no profiler, no restart, R2 first then R1, the same prompt r5, no cached nodes;
the hires videos of both are bit-identical, 113 of 113 frames):

| node | R2, RIFE VFI (Unica) | R1, RIFE VFI | |
|---|---|---|---|
| 1051 RIFE | **40.7 s** (util 57 %, 151 W, VRAM ≤ 13.7 GiB) | 138.1 s (util 21 %, 76 W, VRAM ≤ 5.5 GiB) | **3.4×, −97.4 s** |
| 984 VideoCombine | 50.3 s | 76.1 s | −25.8 s |
| 981 hires decode | 31.0 s | 41.6 s | −10.6 s |
| 1049 Nomos | 19.4 s | 21.0 s | |
| 977 I2V encode | 30.7 s | 18.3 s | +12.4 s, a one-off |
| total | **608.0 s** | 729.8 s | −121.8 s |

Free RAM in the hires window: 2.0 GiB in R2, 0.86 GiB in R1.

Against the prediction:
- the RIFE node 140 s → ≤ 50 s: held, 40.7 s. The GPU went from 21 % to 57 % busy: the loop no longer stands between
  the kernels.
- VideoCombine ≥ 5 s less with the contiguous output: held in direction (−25.8 s), but the size is not RIFE's alone:
  R1 ran with less free RAM, and VideoCombine on identical frames moved 46 → 76 s between earlier renders too. Not
  credited to the node until measured on its own.
- The render: −121.8 s, of which −97.4 s is the node itself. "wan stable work" went 765–767 s (M4.36 A, CPU encoder,
  original RIFE) → 608 s (encoder on the GPU, RIFE VFI (Unica)).

## M4.38 — 0.1.0b1 for PyPI

Decision: the first upload to PyPI is a beta, `0.1.0b1`, so that pip installs it only with `--pre` and nobody takes
it for a finished release.

- pyproject: version 0.1.0b1, classifiers (Development Status :: 4 - Beta, NVIDIA CUDA :: 13, Windows, Linux; all
  checked against trove-classifiers), links to the repository, LOG and rigsy.dev/bench.
- README: links made absolute (PyPI renders the README without the repository around it); Use now starts from
  `pip install --pre unica-attention`, with CUDA torch installed first, since pip would otherwise bring a CPU build.
- The wheel only, no sdist: an sdist has no cubins and would need nvcc. The shipped cubins are newer than the last
  change to csrc/ (2026-11-06), and the wheel holds no local paths.
- Checked from a clean venv outside the repository, torch 2.14.0+cu132, nvcc off PATH: "tested on this machine",
  the four presets at 1.58 / 1.67 / 1.66 / 19.26 %, bit for bit as in M4.27.

**Published** 2026-11-08 by the GitHub release v0.1.0b1 (pre-release) through `.github/workflows/publish.yml`
(trusted publishing, environment `pypi` with a required reviewer and only `v*` tags allowed). PyPI holds the very
file checked above: 2 202 215 bytes, sha256 ca1a0919…a0d3f8. `pip install --pre unica-attention` into a clean venv
with torch 2.14.0+cu132: "tested on this machine", the four presets at 1.58 / 1.67 / 1.66 / 19.26 %.

## M4.39 — small calls run whole: the reserve split image models into one-head pieces

**Where it came from.** An A/B of the native node on the image workflows of the Rigsy site (ComfyUI app's agent; the
site's own graphs, 1024², seed 42, KSampler time, sdpa / unica / unica / sdpa after a warm-up; accurate, per token,
reserve auto). SDPA here is ComfyUI's default, the site runs no SageAttention.

| workflow | SDPA, s | unica, s | sage2, s | PSNR unica / sage2 vs SDPA, dB |
|---|---|---|---|---|
| Qwen-Image-Edit | 6.77 / 18.60 (desktop in use) | **5.76 / 5.82** | 7.06 | 26.3 / 27.7 |
| Z-Image, 25 steps | 13.88 / 14.03 | 17.69 / 11.61 | 14.20 | 28.2 / 3.4 (black image) |
| Z-Image finetune, 8 steps | 5.36 / 4.92 | 5.56 / 5.99 | 5.94 | 26.9 / 32.2 |
| Z-Image + Fun ControlNet, img2img | 10.64 / 10.71 | 11.02 / 13.36 | 18.21 | 32.8 / 35.4 |
| Moss, 30 steps | 10.94 / 11.12 | **19.07 / 18.12** | 14.74 | 24.4 / 24.2 |

- Moss 70 % slower, in both renders. M4.26 had it level.
- Z-Image's two unica renders are bit-identical images, yet 17.7 and 11.6 s; 11.6 s is M4.26's result.
- sage2 here is `sageattn_qk_int8_pv_fp16_cuda`, and on Z-Image it gave a black image: its P·V accumulates in fp16,
  and Z-Image's two tokens with extreme V (M4.22) overflow it. Unica per token is the fix for exactly that case.
- SDXL (heads of 64): every call passed on, as it should. The image differed from SDPA (30.9 dB), but the only SDPA
  image was the cold first render, and the img2img graph's cold render differs from its own repeats too (32.8 dB).
  Not settled without a warm SDPA render.

**The explanation.** Moss loses ~7.6 s over ~1260 self-attention calls: ~6 ms per call, ten times what the kernel
takes on a call this short. `heads_per_call` (M4.31) sizes a call against the driver's free VRAM minus the reserve
(auto: 1.6 GiB). ComfyUI's dynamic VRAM keeps the card filled with staged weights, so the free VRAM can sit below the
reserve, and then a call runs one head at a time: 24 calls of quantization, kernel and Python instead of one. That
fits the scatter too: the same math, a different number of launches, set by what happened to be free.

But the reserve is there for WAN's big calls (2.3 GB of buffers at 75 k tokens), not for these: a Moss call holds
~150 MB, a fifth of what splitting is meant to keep free.

**The change.** A call whose temporary buffers fit in a quarter of the reserve (`SMALL_CALL_FRACTION`, 400 MB on
16 GB) and in the free VRAM runs whole. Bigger calls are sized as before; a call that does not fit in what is free
still goes in pieces. Option chosen over a smaller automatic reserve, which would weaken the guard for WAN as well.
`tests/test_pieces.py` checks both sides of the rule; `bench/pieces_cost.py tokens heads batch` times a shape whole
against one-head pieces.

**Hypothesis, before measuring:**
1. Microbench, Moss-sized call (2 × 4096 tokens, 24 heads), per token: one-head pieces cost +4…6 ms per call over
   the whole call (~0.2–0.25 ms per extra piece).
2. With `UNICA_LOG_VRAM=1` on the old code, most of Moss's calls on the site graph run at fewer than 24 heads, with
   the driver's free VRAM below ~1.7 GiB.
3. Moss on the site graph with the change: 18.6 → ≤ 11 s, level with SDPA or better; Z-Image's two unica renders
   both near 11.6 s.

**Microbench** (`bench/pieces_cost.py`, fp16 in ComfyUI's layout, ComfyUI idle on the same card; median of 4 rounds
× 5): the whole call against pieces of one head.

| shape | per token | per channel | outliers | fast |
|---|---|---|---|---|
| 2 × 4096, 24 heads (Moss-sized) | 2.58 → 6.13 ms (**+3.55**) | 2.45 → 7.88 (+5.43) | 2.97 → 18.63 (+15.66) | 2.45 → 27.87 (+25.42) |
| 2 × 4128, 30 heads (Z-Image) | 3.18 → 7.20 (+4.01) | 3.15 → 9.15 (+6.00) | 3.77 → 21.84 (+18.08) | 3.09 → 33.29 (+30.20) |

- Hypothesis 1 missed low: +3.55 ms per call, not +4…6 (~0.15 ms per extra piece, not 0.2–0.25). Over Moss's ~1260
  calls that is ~4.5 s of the 7.6 s lost, if every call was split and the site's calls have this shape; how many
  calls Moss makes per step (one batch of 2 for CFG, or two of 1) is not known here. The render test decides.
- Splitting costs far more in fast and outliers: per piece they launch more (fast's q̄ · k table, outliers' top-k), up
  to 10× the whole call.
- `tests/test_pieces.py`: 12 passed on the GPU.

**Renders on the site graphs** (ComfyUI app's agent, ComfyUI restarted on this branch with `UNICA_LOG_VRAM=1`;
1024², seed 42, KSampler time, sdpa / unica / unica / sdpa after a warm-up):

| workflow | SDPA, s | unica before, s | unica now, s |
|---|---|---|---|
| Moss | 10.72 / 10.70 | 19.07 / 18.12 | **11.34 / 11.34** |
| Z-Image | 12.94 / 13.71 | 17.69 / 11.61 | **11.71 / 11.62** |
| SDXL (raehoshi) | 10.09 / 10.10 | — | 10.10 |

What the sizing log showed:

- **Moss**: 2640 calls per render. The self-attention calls are 4131 and 4117 tokens, 1260 of each, so 30 steps ×
  42 blocks run twice: CFG goes as two batches of 1. The driver's free VRAM in them was 1.26–1.62 GiB, median 1.48,
  against a reserve of 1.59. **The old code would have split all 2520**, and now none are split.
- **Z-Image**: free VRAM ~3.1 GiB (2.96–4.59), so the old code would have split **0 of 1600**.

Against the hypothesis:

1. The microbench missed low (above). With the call count now known, the render agrees with it: −7.3 s over 2520
   calls of batch 1 is 2.9 ms per call, against +3.55 ms measured for a call of batch 2.
2. **Held.** Moss ran in pieces in every call, and the free VRAM sat just under the reserve, as predicted.
3. **Half held.**
   - Moss 18.6 → 11.3 s, the scatter gone. But the hypothesis was ≤ 11 s and level with SDPA. It is still +6 %
     slower than SDPA. That fits M4.26 (Moss +3 %): its calls are short, batch 1 and ~4.1 k tokens, and the
     quantization around the kernel does not pay off there.
   - Z-Image: both renders near 11.6 s (11.71 / 11.62), 12 % faster than SDPA. **But the explanation was wrong
     for Z-Image.** Its calls were never split, so the earlier 17.7 s render came from something else, most likely
     the desktop in use during that run (SDPA's 18.6 s on Qwen came from the same run).

SDXL settled: warm SDPA and unica are bit-identical, and every call passed on with `head size 64`. The 30.9 dB earlier
was the cold reference render.

**Result:** on the site's image models unica is faster than SDPA on Qwen-Image-Edit (−15 %) and Z-Image (−12 %),
slower on Moss (+6 %), and level on the Z-Image finetune and img2img (single-render scatter there, not re-run). SDXL
is untouched.

**0.1.0b2.** The change goes out as the second beta: version 0.1.0b2 and the README by signature (the measured models
described by tokens × heads and what sets their values apart, the names left to bench/comfyui/models.md). The wheel
was built fresh by build_wheel.py, the four cubins as before. It was checked from a clean venv outside the repository,
with torch 2.14.0+cu132 and nvcc off PATH: "tested on this machine", the four presets at 1.58 / 1.67 / 1.66 / 19.26 %,
the same as b1. 2 202 803 bytes, sha256 9d3f2f2c…c4548d. The cubins' line info names the source folder on the build
drive and the CUDA toolkit's folder, but no user folder.

**Published** 2026-11-09 by the GitHub release v0.1.0b2 (pre-release) through the same workflow. PyPI holds the very
file checked above (sha256 9d3f2f2c…c4548d). For a few minutes the simple index behind PyPI's CDN still listed only b1,
and pip could not find b2. `pip install --pre unica-attention==0.1.0b2` into a clean venv: SMALL_CALL_FRACTION present,
"tested on this machine", the four presets at 1.58 / 1.67 / 1.66 / 19.26 %.

## M4.40 — the hires decode: why it is slow in an old process (T9)

The hires decode 981 (WanVideoDecode, 113 frames of 768 × 1360, untiled) has taken two very different times:

| renders | 981 |
|---|---|
| M4.36 A, B, A2 (an old process, `/free` before each) | 80.8, 88.6, 83.6 s |
| M4.37 r4 (profiled) | 69.1 s, VRAM 15.6–15.9 GiB, +1.5 GiB shared, 98 % busy at 133 W |
| M4.37b right after a restart, then the clean A/B | 26.4, 33.8, then 31.0, 41.6 s |

T9 said: the VRAM is held by staged models and torch's cache, the decode spills to shared memory, and freeing the
VRAM before 981 fixes it. Before anything is changed, the workflow and WanVideoWrapper's code weaken that:
- WanVideoDecode already calls `soft_empty_cache()` before and after the decode, so torch's cache is not it;
- the hires sampler 976 has `force_offload` on and block swap keeps all 40 blocks in the RAM, so the 14B model
  leaves the GPU after sampling;
- M4.36's renders peaked at 10.5 GiB of VRAM for the whole render, and 981 still took 81–89 s.

So a purge before 981 (the workflow already has two PurgeVRAM nodes, both after it) may do nothing. The mechanism is
checked first, apart from ComfyUI: [bench/hires_decode.py](bench/hires_decode.py) decodes the hires latent's shape
([1, 16, 29, 96, 170], random values: a convolution's time does not depend on them) with WanVideoWrapper's own VAE
code, while a ballast tensor takes a set amount of the VRAM. Windows' sysmem fallback lets an allocation go past the
VRAM into shared memory instead of failing, which is the suspected mechanism. ComfyUI leaves `cudnn.benchmark` off
unless `--fast autotune`; the bench imports ComfyUI's model management, so it runs the same way.

Hypotheses, written before measuring:
- H1: in a fresh process, without ballast, the decode takes 26–34 s, as right after a restart; torch's peak is
  8–12 GiB.
- H2 (T9's mechanism): the time stays flat while the ballast plus the peak fit in the free VRAM, and rises ≥ 2×
  (≥ 60 s, as in M4.36–M4.37) once ~1.5 GiB of it lands in shared memory.
- H3, the alternative: if the time rises before anything overflows, the spill is not the cause, and cuDNN choosing
  algorithms with a smaller workspace when free VRAM is short becomes the suspect (checked then by the kernel names
  under ncu, fast against slow).
- H4: tiled (the node's defaults, 272 / 144 px) peaks at ≤ 4 GiB, stays flat across the ballasts, and costs ≤ 1.3×
  the untiled time without ballast.

**Result** (bench/hires_decode.py, ComfyUI idle, the desktop holding ~2 GiB: 14.57 GiB free at the start):

| decode | peak | over the free VRAM | time |
|---|---|---|---|
| whole | 15.15 GiB | 0.58 GiB | 102.7 s (98.7, 97.3 with the real latent) |
| whole, ballast 4 GiB | 15.15 | 4.58 | 549.6 s |
| whole, ballast 6 GiB | 15.15 | 6.58 | 699.1 s |
| tiled 272 / 144, ballast 0, 6, 10 GiB | 1.45 | 0 | 70.4, 70.0, 68.5 s |

During the spilled whole decode the GPU showed 99 % busy at 71 W and 32 °C; the tiled one ran at 232 W. That is the
machine-level picture of 981 in M4.37 (98 % at 133 W): the SMs are scheduled but wait on PCIe for the activations that
live in shared memory. The larger runs (8–12 GiB of ballast) were stopped: the time only grows, ~100 s per GiB over.

- H1 missed: the whole decode is not 26–34 s in a fresh process, and its peak is not 8–12 GiB. It needs 15.15 GiB
  by itself, so on a 16 GiB card with a desktop it overflows even with nothing else loaded.
- H2 held, much steeper than written: 0.58 GiB over already costs ~3× (98–103 s against ~29 s that fit), and the
  time keeps growing with the overflow. The fast renders after a restart (26–34 s) were the ones where it happened
  to fit. "An old process" was a proxy: what counts is how much of the VRAM everything else holds at that minute,
  the desktop included. Freeing before 981 cannot fix it: WanVideoDecode already empties torch's cache, the model is
  already off, and the decode alone does not fit.
- H3: nothing to separate. The slow runs all overflowed; the tiled ones never did and stayed flat.
- H4 half held: the peak 1.45 GiB and flat across the ballasts, but 68–70 s, 2.3× a decode that fits, not ≤ 1.3×.
  The default tiles (34 × 34 latent, stride 18 × 16) compute 3.2× the frame's area in their overlaps.

**Quality of the tilings**, on a real latent: m437c R2's hires video (113 portrait frames, 1360 high, 768 wide)
encoded by the same VAE, then decoded whole (the reference) and tiled. The whole-video PSNR, and the worst 16-pixel
band of columns and of rows, where a seam would gather (rounding to 8 bits alone is ~59 dB):

| tiling (tile / stride, px) | peak | time | PSNR | worst column / row band |
|---|---|---|---|---|
| 272 / 144 (the node's default) | 1.44 GiB | 64.5 s | 50.6 dB | 47.8 / 46.1 dB |
| 512 / 256 | 4.01 | 50.9 | 53.2 | 51.5 / 50.0 |
| three tiles, frame-wide, 512 high, 88 px overlap | 7.67 | 32.0 | 54.8 | 49.7 / 50.8 |
| two tiles, frame-wide, 768 high, 16 px overlap | 11.30 | 29.1 | 57.0 | 54.8 / 53.5 |
| two tiles, 176 px overlap | 11.30 | 28.7 | 57.0 | 54.8 / 53.5 |

- The default's error shows a faint grid of seams at 8× brightness (build/m440_diff8_272_144.png); two tiles show
  only the contours, no seam (m440_diff8_two_tiles.png). The memory that tiling spoils the picture is the small
  default tiles.
- The overlap of the two tiles does not change the result: their worst bands are not at the seam.
- WanVideoWrapper's tiled decode adds up the tiles in `values` and `weight` of the latent's dtype (bf16 here) and
  divides in bf16; with 45 overlapping tiles that rounding is a likely part of the default's error. Not measured
  apart.
- Two frame-wide tiles take as long as a whole decode that fits (~29 s) at 3.85 GiB less: the margin under the 14.6
  GiB a desktop leaves is 3.3 GiB instead of −0.6.

## M4.41 — per token behind per channel: the token scales waited on L2

**Not ping-pong.** The first idea for the kernel was FlashAttention 3's ping-pong: keep the two warps of a scheduler
out of phase, so that one runs its softmax while the other issues mma. M3's timeline already measured what it fixes
(H1, bunching): the time with no warp in an MMA phase was 0.078 against 0.125 for independent warps, so the warps
already interleave better than chance. What holds the tensor pipe at ~64 % is the instruction mix: every non-MMA
instruction of the scheduler costs tensor time (`peak_u8s8_mixed`: 8 FFMA per IMMA −19 %, unchanged with 4 warps per
scheduler). On Hopper `wgmma` runs asynchronously once issued; `mma.sync` takes the issue port per instruction, so
reordering the warps does not lower the count. Expected 0…−5 % for a rebuild to 8 warps per block (W8 was slower in
M3). Not done. The lever by that model is fewer instructions per tile, and the largest gap is T8: per token behind
per channel.

**Measured** (compare.py, one process, 6 alternating rounds, 40 × 34,020): kernel 82.7 ms per channel, 93.1 ms per
token, **+12.6 %** (M4.33: +12 %).

Hypotheses, written before ncu:
- H1: per token 10–14 % slower than per channel in one process. **Held**, +12.6 %.
- H2: the extra cycles are stalls, mostly long scoreboard (the token scales' LDG): ≥ half of the added stall cycles.
  **Held**, 71 %.
- H2b: if wait and math pipe grow instead, it is the schedule. Not so.

SASS of the loop (4 tiles in its body, per tile): per channel 578 instructions, per token 641 (+63): FADD +24, FFMA
+24, LDG +6, IDP −6, 0.5 LDL. Registers 235 and **255**, stack 8 and 24 bytes (12 bytes spilled). By the M3.9 model
(~1 cycle per non-MMA instruction) +63 instructions are +2.2 % of a ~2808-cycle tile: **the model explains a fifth of
the gap.** ncu (4 heads × 34,020) shows why: cycles +14.3 %, instructions +15.7 %, but cycles per issued instruction
the same (5.72 / 5.65), and the stall cycles (per-issue ratio × instructions):

| stall | per channel | per token | |
|---|---|---|---|
| long scoreboard | 0.19 | 1.44 | **+1.25** |
| selected (issuing) | 2.12 | 2.45 | +0.33 |
| wait | 3.07 | 3.35 | +0.28 |
| mio throttle | 0.34 | 0.59 | +0.25 |
| barrier | 1.21 | 0.81 | −0.40 |
| all | 12.08 | 13.85 | +1.77 (+14.7 %) |

Per instruction (ncu SourceCounters) the long-scoreboard samples of per token sit on: the FFMA that use the token
scales (41 %), their `LDG.E.128.CONSTANT` 38–46 instructions earlier; the FMUL that uses `k_scale` of the tile
(14.5 %), its LDG 26 instructions earlier; two IMMA (29 %) without a load as producer. The comment said "issued before
Q·Kᵀ(t + 1)", but a tile's token scales are 8 float4, 32 registers per thread, and at the 255-register cap ptxas moved
the loads next to their use: ~40 instructions at ~5.7 cycles each is ~230 cycles, less than L2's latency.

**Change** (`UNICA_TOKEN_COPY`, default 1; 0 is M4.23): a tile's token scales are 512 contiguous bytes in v_token
(padded to whole tiles like V8). Thread 0 copies them into shared memory with `cp.async.bulk`, on V's mbarrier, in the
same step as V(t + 1); the softmax reads them with LDS from the buffer of the tile. The values and the arithmetic are
the same.

Hypotheses, written before measuring:
- output bit for bit as TOKEN_COPY=0;
- per token 93 → 85–88 ms (−5…−9 %), long scoreboard down to per channel's level, ≤ 240 registers, no LDL;
- risk: in M4.21 each extra bulk copy of fast cost ~2 ms; here it rides on V's existing copy.

**Result:**

| | TOKEN_COPY=0 | TOKEN_COPY=1 | predicted |
|---|---|---|---|
| output | | bit for bit, 6 lengths (40 … 34,020, partial tiles, an outlier token) | held |
| kernel, 40 × 34,020 | 91.7 ms | **89.5 ms (−2.4 %)**; a second run 91.5 against per channel's ~81.5 | 85–88 ms: **missed** |
| registers, spills | 255, 12 / 16 bytes | 255, **0** | ≤ 240: missed; no LDL: held |
| shared memory per block | 46,104 bytes | 47,128 (2 blocks per SM still fit) | |
| loop SASS per tile | | 6 LDG.128 → 6 LDS.128, +8.5 to issue the copy | |
| active cycles (ncu) | 20.89 M | 20.13 M (−3.6 %) | |
| long scoreboard | 1.44 | **0.20** | held |
| barrier | 0.81 | **1.41** | |

- The wait for L2 is gone, but most of it came back at the tile's `__syncthreads`: the warps reach the barrier at
  different times and wait for the slowest. As in M3.19, removing one stall shows the next.
- What is left of the gap (per channel 81.5 against ~89.5–91.5 ms) is per token's arithmetic: +15 % instructions,
  an FFMA and an FADD per key for log₂ s and the denominator weighted by 1 / s. The data no longer waits.
- Per channel's SASS of `attention_int8pv_tma` is identical to the shipped cubin's. The 56 tests pass.

**Trial: 1 / s from log₂ s** (`exp2(−log₂ s)` on the SFU instead of reading it; it was meant to free 16 registers,
which the copy already made moot). Hypothesis: 0…+2 % slower, rel L2 the same to 0.01 %. Measured +0.9 % (92.3
against 91.5 ms, inside that run's 6 % spread), rel L2 1.58 % in both: held, no gain. Not kept.

The shipped per-token cubin (unica/cubins) is from before this change; the next wheel rebuilds it.

## M4.42 — frame-wide tiles for the hires decode, in the render

M4.40 measured the hires decode apart from ComfyUI: whole, it needs 15.15 GiB and overflows a 16 GiB card that also
holds a desktop; two frame-wide tiles take ~29 s at 11.3 GiB, three take ~32 s at 7.7 GiB, both at 55–57 dB against
the whole decode. Now the same in the render, by the ComfyUI agent, "wan stable work" as in M4.37's clean A/B
(encoder on the GPU, RIFE VFI (Unica)), the workflow file untouched (981's settings set in memory).

Plan:
1. One full render, untiled, to warm the process up; then the same graph queued again with only 981's settings
   changed: untiled, two tiles (896 / 768 px, stride 464 / 752), three tiles (896 / 512, stride 464 / 424), untiled
   again. ComfyUI's cache reruns only 981 and what follows it. For each: 981's time, the VRAM in use and the WDDM
   shared memory at its start and at peak, the GPU's power; frames 0, 56 and 112 of 981's output saved as PNG.
2. Two full renders with the tiling chosen from step 1, in the same process.

Hypotheses, written before the runs:
- H1: the untiled 981 is fast (≤ 35 s) only when the free VRAM at its start covers its 15.15 GiB peak, and slow
  (≥ 60 s) when ≥ 0.5 GiB of it lands in shared memory; its time follows that overflow, not the age of the process.
- H2: two tiles: 29 ± 4 s in every run, three tiles 32 ± 4 s, shared memory flat during 981.
- H3: the tiled frames against the untiled ones (PNG, 8 bits): PSNR ≥ 50 dB (M4.40: 54.8–57.0 over the video).
- H4: Nomos, RIFE and VideoCombine take the same time within their run-to-run noise: 981's output is the same size.
- H5: in the two full renders 981 takes ≤ 33 s, and each render is 0–55 s shorter than an untiled one of the same
  process (0 when the untiled decode happened to fit, ~55 s when it did not).

**Step 1** (ComfyUI agent, seed 42, suffix r6; W a full render, then U1, T2, T3, U2 re-queued with only 981
changed; VRAM is nvidia-smi's, desktop included, ~1.4 GiB idle; files build/m442_*):

| run | 981 | VRAM start → peak | shared start → peak | power mean / max | Nomos | RIFE | VideoCombine |
|---|---|---|---|---|---|---|---|
| W, untiled, full render | 28.0 s | 14.98 → 15.54 GiB | 21.65 → 23.25 GiB | 219 / 265 W | 21.3 s | 54.4 s | 28.9 s |
| U1, untiled | 25.7 s | 1.74 → 15.39 | 21.21 → 21.21 | 226 / 252 | 18.3 | 38.4 | 39.7 |
| T2, two tiles | 31.5 s | 3.72 → 13.59 | flat | 238 / 253 | 16.0 | 28.4 | 25.1 |
| T3, three tiles | 34.8 s | 3.69 → 9.92 | flat | 239 / 251 | 16.5 | 27.2 | 22.7 |
| U2, untiled again | **47.5 s** | 3.67 → 15.48 | 21.20 → **22.85** | **155** / 188 | 19.8 | 28.1 | 24.2 |

PSNR against U1 (frames 0 / 56 / 112): W and U2 bit-identical; T2 55.8 / 53.2 / 52.7 dB; T3 54.4 / 51.4 / 51.2 dB.
In U1 the high-noise sampler 934 ran again after 981 (likely dropped from the cache under RAM pressure, RSS 47.5 GiB);
981's numbers are not touched by it.

- H1 half held. The untiled decode fits in ComfyUI more often than M4.40 implied: it peaked at 15.4–15.5 GiB of the
  card with the desktop, so its own need in ComfyUI is ~14 GiB, not 15.15 (the bench's figure is torch's reserved
  peak under the default allocator; ComfyUI's allocator and the smaller desktop of this run leave it under the
  card). It is close to the edge either way: U2, the same decode as U1 to the bit, spilled 1.65 GiB and took 47.5 s
  at 155 W. Shared growth alone does not mark a slow run: in W it grew by 1.6 GiB and 981 still took 28.0 s at
  219 W, so WDDM moved something else out (staged weights) and the decode stayed in the VRAM. What decides is whose
  memory is demoted, not the age of the process.
- H2 held: two tiles 31.5 s, three 34.8 s, shared flat, power 238–239 W.
- H3 held: 52.7–55.8 and 51.2–54.4 dB.
- H4 held: Nomos, RIFE and VideoCombine move with the run, not with the tiling (RIFE 54 s in W, 27–38 s after).

Every untiled 981 measured so far (M4.36–M4.42): 80.8, 88.6, 83.6, 69.1, 26.4, 33.8, 31.0, 41.6, 28.0, 25.7, 47.5 s;
mean 50.6 s, half of them ≥ 41 s. Two tiles cost 4–6 s when the untiled decode fits and save 16–57 s when it does not.

## M4.43 — the hand's blur in motion: SageAttention 2 instead of unica, and RIFE at scale 0.5

In r6 of M4.42 the hand blurs while it moves. Is it the kernel (quantized attention), RIFE ×5, or WAN itself? The
ComfyUI agent rendered "wan stable work" as W in M4.42 (seed 42, prompt r6, 981 untiled, RIFE VFI (Unica) rife49,
×5, fast_mode, no ensemble) with only the attention changed: the loaders' "sageattn" (SageAttention 2.2.0), the
unica node off. Then the same graph re-queued with RIFE's scale_factor 0.5. Sharpness is the variance of the
Laplacian over a 320 px crop around the strongest motion of a frame pair.

Hypotheses, written before the runs:
- H1: the hand's sharpness with SA2 is within ±10 % of U1's (unica): the blur is WAN's. SA2 > 15 % sharper would
  mean the kernel adds blur.
- H2: with scale_factor 0.5 the in-betweens lose ≤ 12 % sharpness against their end frames instead of 20–25 %, and
  RIFE takes no longer.

| | U1 (unica) | SA2 | SA2 / U1 |
|---|---|---|---|
| frames 73 / 74, U1's crop | 692 / 717 | 740 / 810 | 1.07 / 1.13 |
| U1's 20 strongest-motion crops, pairwise | | | median 1.13 (0.99 … 1.26) |
| whole frame, every 8th frame, median | 654 | 650 | **0.99** |
| the static third of the frame | 16.5 | 15.0 | 0.91 |

The two videos are different renders (PSNR 24.7 dB between them): another attention changes the sampling path.

- H1: **neither held nor refuted by the number.** In the moving crops SA2 reads 13 % higher, between the two
  thresholds; over the whole frame the two are level (0.99), in the static part SA2 is lower (0.91). The crops show
  why the moving number is not sharpness: both hands are smeared the same way, and SA2's smear has streaks (high
  frequencies, which raise the Laplacian). The blur is in SA2's frames as much as in unica's: it comes from WAN
  (16 fps, 4 lightx2v steps), not from the kernel. Files: build/m443_hand_U1_vs_SA2.png (U1 73, 74, SA2 73, 74).
- H2: **wrong.** The in-betweens lose 24.7 % at scale 1 and 22.1 % at 0.5 (median of the 20 strongest-motion pairs);
  the end frames themselves are already blurred, and RIFE interpolates between blurred frames.
  build/m443_rife_*.png. The in-betweens of the two scales differ by 33.5 dB.
- RIFE took 101.5 s at scale 1 and 25.2 s at 0.5, but step 1 ran with the system RAM full (63.1 of 63.5 GiB,
  ComfyUI's RSS 47.6 GiB, up to 175k page reads per second); in M4.42 scale 1 took 27–41 s. The RAM also evicted
  ComfyUI's cache again: in step 2 the high-noise sampler 934 and the decode 981 ran again (981 87 / 83 s against
  25.7 s in U1). The times of this run are not comparable; scale 0.5's speed needs its own A/B.

## M4.44 — hypotheses for less arithmetic at the same error (written before any of them is tested)

Where the kernel stands (accurate, per channel, 40 × 34,020): ~81.5 ms against a tensor floor of ~53.5 ms
(2 × 11.85 TOP of INT8 at 443 TOPS). M3.8: every FMA- or ALU-pipe instruction past ~4 per IMMA takes ~0.7–0.9 cycle
from the tensor pipe; I2FP and MUFU are free. Executed per warp and tile (M3.18): 467, of them 64 IMMA, 98 FFMA (32
for the exponent's argument, 64 for the flush `out += β · acc`), 38 FMNMX, 16 F2IP, 8 IDP. The flush cannot be thinned
in the kernel as it is: P is quantized against each tile's own max (every tile gets all 256 steps), so every tile has
its own β, and int32 accumulators kept across tiles would need 64 more registers at a 255 cap (M3.21).

Correction to M4.41: the loop body holds **3** tiles, not 4 (192 IMMA, 64 per tile: 32 for Q·Kᵀ, 32 for P·V). Its
per-tile static counts were divided by 4: per channel 770 static instructions per tile, not 578; per token +95, not
+63. The M3.9 model then explains +3.0 % of the +12.6 %, not +2.2 %: still about a quarter. The static count is larger
than the 467 executed because the masking (32 FSEL) and the rescale (64 FMUL) sit in branches a tile usually skips.

**Checked at once by compiling (sm_120a, nvcc 13.2):**
- C1: Blackwell's packed FP32 (`fma.rn.f32x2`, `add.rn.f32x2`, `mul.rn.f32x2`), which would halve the 98 FFMA. PTX
  accepts them for sm_120a, but ptxas emits two scalar FFMA / FADD / FMUL each; sm_100a gets FFMA2 / FADD2 / FMUL2.
  **Not on this card**, like the three-input `max.f32` (M3.18).
- C2: a three-input integer max (`max.s32 d, a, b, c`) for the tile max on the int32 scores: rejected by ptxas for
  sm_120a and sm_100a alike. Not in PTX.

The error budget: "at the same error" means each model's real median stays below SageAttention 2's (M4.26: WAN 1.51,
LTX 0.58, Qwen 1.81, Z-Image 2.07, Moss 2.08 %) and at most +0.2 pp above today's kernel on the same dumps.

| # | hypothesis | attacks | prediction | test |
|---|---|---|---|---|
| H1 | a warp skips the softmax and P·V of a tile when all its 16 rows' tile max lies below their running max by more than log₂ τ (M4.7's H2, now at coarser τ = 2⁻⁶, 2⁻⁸, 2⁻¹⁰ and per warp, not per block of 64) | tensor pipe (the P·V half), FMA pipe (flush, exponent), SFU | very uncertain. M4.7 at τ = 2⁻¹², 16 rows: median 0.86 %, mean 6.1 % of the pairs. At 2⁻⁶: mean **10–25 %** of warp-tiles on WAN, error +0.05…+0.3 pp. A skipped warp-tile saves ~55 % of its issue: kernel −3…−10 % where the share holds | `negligible_tiles` for the shares, `emulate(skip_below=τ)` for the error, on the dumps of all five models |
| H2 | a block-sparse mask predicted from pooled Q and K (the mean of a block's 64 queries · the mean of a tile's 64 keys) skips Q·Kᵀ as well, as SpargeAttn does | all three pipes for the skipped pairs | unlikely on this data: M4.7 found 0.05 % of (64-row block, tile) pairs negligible at 2⁻¹² (median). An oracle (exact scores, the row's final max) bounds any predictor: if it gives < 15 % at 2⁻⁸ on WAN, H2 is not pursued | the oracle share first; the pooled predictor's recall only if the oracle passes |
| H3 | per token: the denominator weighted by 1 / s with `dp2a` (two u16 weights × two u8 P̂ per instruction, 16 per tile), not 32 FFMA | FMA pipe | −32 FFMA, +16 IDP, +2 LDS per tile: per token **−1.5…−3 %** (89.5 → 87–88 ms); rel L2 within ±0.01 pp on random data and on Z-Image's outlier tokens (weights scaled per tile: 2⁻¹⁷ steps) | compare.py, 6 alternating rounds; tests/test_v_token.py; SASS: FFMA −32, IDP +16 per tile, no LDL |
| H4 | measure only: the share of warp-tiles whose β < 2⁻⁴ in all 16 rows, i.e. tiles that weigh ≤ 1/16 of the row and could take a cheaper P·V (FP4, 2× the rate) | tensor pipe | 10–30 %. Below 25 % not worth it: a second V format per stage costs ~9 KB of shared memory, and the second block per SM (47 KB each now) | the same pass as H1 |

Order: H1, H2's oracle and H4 in one offline pass over the dumps (the emulator, no ComfyUI). H3 independently, in
the kernel.

**H1, H2 (oracle), H4 measured** (the emulator as the kernel, `TODAY` = per channel, a warp of 16 rows deciding
alone; 6 dumped calls per model spread over its run, 2 heads each; `build/m444_skip.txt`). Shares are of warp-tiles,
median (mean) over the calls; the error column is the median rel L2 and, in brackets, the worst rise over today on
any call:

| model | today | SA2 | τ = 2⁻⁶ | τ = 2⁻⁸ | **τ = 2⁻¹⁰** | β < 2⁻⁴ (H4) | oracle, 64 rows, 2⁻⁸ (H2) |
|---|---|---|---|---|---|---|---|
| WAN | 0.933 | 1.078 | 22 (27) %, 1.743 (+4.21) | 9 (18) %, 1.062 (+0.71) | **3 (11) %, 0.962 (+0.08)** | 32 % | 6.1 % |
| Z-Image | 12.18 | 2.517 | 8 (28) %, 12.42 (+0.42) | 2 (23) %, 12.24 (+0.10) | **0.3 (18) %, 12.19 (+0.02)** | 21 % | 0.04 % |
| LTX | 0.451 | 0.577 | 0.2 (2) %, 0.751 (+0.71) | 0 (1) %, 0.451 (+0.03) | **0 (0.3) %, 0.451 (+0.003)** | 4 % | 0 % |
| Qwen | 1.465 | 2.035 | 52 (46) %, 2.349 (+8.89) | 32 (30) %, 1.636 (+1.72) | **18 (17) %, 1.479 (+0.03)** | 69 % | 21 % |
| Moss | 1.407 | 2.136 | 35 (36) %, 1.548 (+0.15) | 28 (29) %, 1.446 (+0.04) | **20 (21) %, 1.414 (+0.007)** | 44 % | 42 % |

(Z-Image's "today" is per channel, which its outlier tokens break, M4.25; the shipped preset for it is per token or
outliers. The skip's own rise is what this table compares.)

- **H1: held at τ = 2⁻¹⁰, wrong at 2⁻⁶.** Predicted 10–25 % of WAN's warp-tiles at 2⁻⁶ for +0.05…+0.3 pp: the share
  came (27 %), the error did not (+4.2 pp worst). Each skipped tile weighs < τ of the row, but hundreds of them add
  up: the bound that matters is the skipped mass of a row, not of a tile. At 2⁻¹⁰ every model stays inside the budget
  (worst +0.08 pp, WAN), and a warp skips 11 % (WAN), 17 % (Qwen), 18 % (Z-Image), 21 % (Moss) of its tiles on average,
  0.3 % on LTX. **The image models skip as much as video, and more evenly** (Qwen and Moss: median ≈ mean); WAN's
  skips sit in a few very sharp calls (call 42: 41 %, call 70: 21 %, most others ≤ 6 %).
- M4.7 closed this idea on WAN at 2⁻¹² per block of 64 rows (median 0.05 %). Per warp, at 2⁻¹⁰ and on all models it is
  open. In the kernel a warp that skips still meets the tile's barrier; the gain counts only if the other block on
  the SM (2 per SM) takes the issue slots it frees. That is the kernel test.
- **H2: not pursued.** The oracle (exact scores, the final max) at 2⁻⁸ gives WAN 6.1 %, below the 15 % set before
  measuring; and 2⁻⁸ is already too coarse for the error by H1. Moss (42 %) and Qwen (21 %) would pass the share, at a
  τ that H1 shows is unsafe.
- **H4: above the predicted 10–30 %** on WAN (32 %), Qwen (69 %) and Moss (44 %). Not built now: a second V format per
  stage costs the second block per SM; noted for later.

**H1 in the kernel** (`SKIP_TILES=n`, default 0; attention_int8pv_tma, V per channel). After the tile max of both
rows, a warp votes; if every lane's two rows lie below their running max by more than n, it skips the exponents,
the packing, the row sum, P·V and its flush, and leaves row_max, row_sum and out as they are (the emulator's
`skip_below` exactly). Q·Kᵀ(t + 1) is already issued above it, so the overlap stays. A copy of online_softmax_u8,
not a flag inside it: the default build's SASS is bit for bit the shipped one (checked). With n = 10: 233 registers
(235 before), no spill; the loop gains per tile 2 FADD, 2 FSETP, 1 VOTE, 1 BRA (static +26 over 3 tiles).
tests/test_attention.py: random data bit for bit with and without; sharp data (> 10 % skipped) within 1e-3 of the
skipping emulator.

Hypotheses, written before timing (the dumps of the table above, the kernel alone on its own inputs, SKIP_TILES 0 / 10
alternating in one process):
- the time follows the skipped share at ~55 % of a tile each: Moss −6…−11 %, Qwen −5…−9 %, Z-Image's sharp calls
  −10…−25 % (its others ~0), WAN −3…−6 % over its calls (most near 0, calls 42 and 70 −10…−20 %), LTX +0.5…+1.5 % (the
  vote with nothing skipped);
- if the per-tile barrier makes a skipping warp wait for the block's slowest, the gains come out at about half: that
  is the finding to look for, not to explain away;
- the kernel's rel L2 rises as the emulator's did, ≤ +0.08 pp on every call.

**H1 in the kernel, result** (`bench/skip_tiles.py 10 6 12`: the dumps' 2 heads repeated to 24, so that a call has a
real call's number of blocks; 6 alternating rounds × median of 20; `build/m444_skip_kernel_24heads.txt`):

| model | on / off, mean (best … worst) | calls with nothing skipped | the most skipping call |
|---|---|---|---|
| WAN | 1.000 (0.925 … 1.032) | +2.8…+3.2 % | call 42, 41 % skipped: −7.5 % |
| Z-Image | 0.986 (0.872 … 1.034) | +3.1…+3.4 % | call 630, 61 %: −12.8 % |
| LTX | 1.032 (1.028 … 1.036) | +2.8…+3.6 % (all) | — |
| Qwen | 0.988 (0.954 … 1.028) | +2.1…+2.8 % | call 228, 31 %: −4.6 % |
| Moss | 0.983 (0.952 … 1.008) | — | call 1260, 31 %: −4.8 % |

The errors are the emulator's: the kernel's rel L2 rises by ≤ +0.08 pp (WAN call 56: 1.043 → 1.124 %; call 70: 0.638 →
0.715 %), 0 on LTX. With 2 heads instead of 24 the small models gained nothing (Moss +2.3 %): less than one wave of
blocks, so the slots a skipping warp frees have no other block to go to.

- **The hypothesis was wrong twice.** (1) With nothing skipped the kernel is **+3 %** slower, not +0.5…+1.5 %. The first
  build voted before the exponents, and the SASS showed why it cost: the branch split the softmax from Q·Kᵀ(t + 1) into
  two basic blocks, and ptxas issued all 32 IMMA of Q·Kᵀ(t + 1) before the exponents instead of between them (the
  order of IMMA and MUFU in the loop: `QEQEEQEEQ…` by default, `QQQ…QVEEEEEE…` with the vote). Moving the vote after the
  softmax (beta < 2⁻ⁿ in both rows is the same test) restored that interleave and changed nothing in time: the cost
  is elsewhere, in +9 MOV per tile (ptxas moves registers around the branch before P·V, 232 → 238 registers) and the
  vote itself, ~17 instructions on 467. (2) A skipped warp-tile saves **~26 %** of a tile, not 55 % (the most skipping
  calls: 61 % skipped → −12.8 % + 3 % = 15.8 %; 41 % → 10.5 %; 31 % → 7.8 %). Half of the gap is that only P·V and its
  flush are skipped now; the other half is the barrier: the warp that skips waits for the block's others.
- **Net: a tie.** −1.2…−1.7 % on the image models, 0 on WAN, +3.2 % on LTX. Not a default (option a of the plan); it stays
  as `SKIP_TILES=n`, default 0, whose default build is bit for bit the shipped one.
- The lever it leaves: the skip pays only when it skips together. A block of 64 rows that skips as a whole (M4.7's
  shares, smaller) frees the barrier too, and a branch that costs nothing when not taken would remove the +3 %.

**H3, emulator first: refuted before the kernel.** `dp2a` needs the denominator over the rounded u8 P̂' (as per channel
sums it since M3.15), with 1 / s_j in 16 bits. The emulator got that as `v_token="rounded16"` (1 / s_j rounded to 16
bits against the largest of its tile); 6 dumped calls per model, per token as the kernel (`v_token="float"`) against
it (`build/m444_h3_emulator.txt`):

| model | float (today), median | rounded16, median | rounded16 − float, per call |
|---|---|---|---|
| WAN | 0.775 % | 0.804 % | −0.03 … +0.11 pp |
| Z-Image | 1.144 % | 1.345 % | +0.01 … **+0.43** pp |
| LTX | 0.434 % | 0.532 % | −0.00 … +0.31 pp |
| Qwen | 1.217 % | 1.306 % | −0.01 … +0.28 pp |
| Moss | 1.134 % | 1.209 % | −0.05 … +0.08 pp |

- The 16-bit weights cost nothing: `rounded16` equals `rounded` (float 1 / s_j, rounded P̂') to 0.001 pp on every call.
  What costs is rounding P' before the sum. A key with a small s_j has a small P' = p · s_j, rounded against the
  tile's largest P' to a few steps or to 0, and the denominator multiplies that back by its large 1 / s_j: the
  rounding error grows by the same factor. Per channel has no such factor. This is why M4.23 summed in float.
- Predicted "within ±0.01 pp": wrong by up to 40×, and over the +0.2 pp budget on Z-Image, LTX and Qwen. Not built.

**H5 (from H1's result): the whole block skips, decided a tile ahead.** H1 lost to two costs: the warp that skips
waits at the block's barrier, and the branch in the middle of the tile costs +3 %. Both go if the block of 64 rows
decides together, before the tile starts. Q·Kᵀ(t + 1) is computed in iteration t, so its tile max can be taken
there; each warp writes "negligible" to shared memory before the barrier that iteration t + 1 has anyway, and after
it the whole block either runs the tile or branches over all of it (exponents, packing, P·V, flush), at the top of
the iteration, where the branch does not split the body. Hypotheses, written before the emulator:
- the share of (64-row block, tile) pairs negligible at 2⁻¹⁰ is about half of H1's per-warp share (M4.7 at 2⁻¹²: 3.1
  against 6.1 % mean): WAN ~5 %, Qwen ~8 %, Z-Image ~10 %, Moss ~10 %, LTX 0;
- a skipped block-tile saves ~55 % of its time (all but Q·Kᵀ and the barrier), with no cost when nothing is skipped:
  kernel WAN −2…−3 %, Qwen, Z-Image and Moss −4…−6 %, LTX 0 ± 0.5 %;
- the error rises less than H1's (fewer tiles skipped).

H5, the emulator (`build/m444_h5.txt`, the same 30 calls, blocks of 64 rows): share of block-tiles skipped at 2⁻¹⁰,
median (mean), and the worst error rise: WAN 0.7 (4.1) %, +0.025 pp; Z-Image 0.0 (14.9) %, +0.011 pp (two calls carry
it); LTX 0 %; Qwen 8.9 (8.5) %, +0.002 pp; Moss 10.1 (11.7) %, +0.002 pp. At 2⁻¹²: WAN 1.4, Z-Image 11.2, Qwen 4.3, Moss
6.5 % (means). **The share held** (predicted WAN ~5, Qwen ~8, Z-Image ~10, Moss ~10 %), and the error rises 3–15× less
than per warp. By the 55 % per skipped tile: WAN −2 %, Qwen −5 %, Moss −6 %, Z-Image −8 % on average (0 on most of its
calls, ~−30 % on the sharp ones). On to the kernel.

**H5 in the kernel** (`SKIP_BLOCKS=n`, default 0, whose build is bit for bit the shipped one). The tile max of tile
t + 1 is taken in iteration t, right after its Q·Kᵀ and mask (moved, not added: the softmax then takes it as
given); each warp votes, every lane writes the warp's byte, and after iteration t + 1's barrier the block reads the
four bytes as one word and either runs the tile as before or does only Q·Kᵀ(t + 2). 231 registers, no spill; the
running path keeps the `QEQEEQ…` interleave. tests/test_attention.py covers both SKIP settings.

`bench/skip_tiles.py 10 6 12 SKIP_BLOCKS` (`build/m444_h5_kernel_24heads.txt`), on / off, mean (best … worst):
WAN 1.009 (0.981 … 1.039), Z-Image 0.997 (0.895 … 1.047), LTX 1.022 (1.005 … 1.036), Qwen 0.997 (0.975 … 1.025), Moss
0.996 (0.955 … 1.025). The errors rise by ≤ 0.026 pp (WAN call 70), 3× less than H1's.

- **Wrong where it mattered:** "no cost when nothing is skipped" did not hold. The calls with nothing skipped are
  +2.4…+4.7 % (LTX, Z-Image's flat calls, WAN calls 7, 14, 28), as with H1. The loop per tile gains ~7 MOV (the
  registers of `scores[later]` brought together after the two paths), a BSSY / BSYNC pair (the reconvergence of the
  if / else itself; writing the vote from every lane instead of lane 0 removed only 2 ISETP), the LDS and compare of
  the vote, address arithmetic: ~15–20 instructions on 467.
- A skipped block-tile saves what was predicted: Z-Image call 630 −10.5 % with ~4.7 % over it, i.e. ~15 % for the
  share it skips; but the average share (4–15 %) does not pay the 3 % toll.
- **What H1 and H5 say together:** in this kernel (231–238 of 255 registers), a data-dependent branch inside the tile
  loop costs ~3 % on its own, through the register moves ptxas adds around it, whichever place it sits. Skipping wins
  only where it skips more than ~20–25 % of the work on average; the real calls skip 4–15 % (blocks) or 11–21 %
  (warps, with the barrier eating half). Both stay as options, default 0. The next lever is not another place for
  the branch but a call that knows in advance: a whole head or call that is sharp could take a separate build.
