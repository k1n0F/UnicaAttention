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

**Consequence:** 103 TFLOPS is 93 % of the measured BF16/FP16 → FP32 `mma.sync` peak (111 TFLOPS).
The tensor-core floor of this precision is 198 ms; at most ~7 % is left on this path.
