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
