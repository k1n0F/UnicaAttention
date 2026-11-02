# UnicaAttention

*Unica*, Latin for "the only one of its kind": an attention kernel made for one GPU, the RTX 5080 (consumer Blackwell, sm_120).

> **Status:** early development. There is nothing to install yet.

## Why one GPU

Consumer Blackwell is not datacenter Blackwell. sm_120 has no `tcgen05` / Tensor Memory and no `wgmma`. Tensor cores are reached only through warp-level `mma.sync`, and each block gets 99 KB of shared memory instead of 228 KB. Kernels tuned for datacenter GPUs do not fit these limits. A kernel built for exactly these limits can.

## Target

Forward attention for video-diffusion inference, at the shapes of WAN 2.2 14B: head_dim 128, 40 heads, ~32 760 tokens (81 frames at 832×480), non-causal, bf16 inputs.

## Baseline

From [rigsy.dev/bench](https://rigsy.dev/bench): shape 1 × 40 × 32760 × 128, bf16, error against FP32 on random inputs.

| Kernel | Time | Speed-up | Error vs FP32 |
|---|---|---|---|
| PyTorch SDPA | 511 ms | 1.0× | 0 % |
| SageAttention 2 (INT8 QK · FP8 PV) | 76 ms | 6.7× | 3.9 % |
| SageAttention 3 (FP4 block-scaled) | 56 ms | 9.2× | 19 % |

UnicaAttention wins if it is **faster than SageAttention 2 at no more than 3.9 % error**, or **more accurate than SageAttention 3 at a comparable speed**. Accuracy comes before speed.

## Approach

- CUDA C++ with inline PTX `mma.sync`. FP8 e4m3 comes first, FP4 e2m1 block-scaled comes later.
- TMA (`cp.async.bulk` + mbarrier) with double buffering.
- nvcc builds a `.cubin`, which Python loads through the CUDA driver API (ctypes). There is no C++ extension to rebuild for each torch version.

## Roadmap

| | Milestone |
|---|---|
| M0 | Build, loader and benchmark harness that reproduce the baseline |
| M1 | Plain, correct flash attention on `mma.sync` BF16 |
| M2 | TMA, double buffering, tile sizes tuned for sm_120 |
| M3 | FP8 path, compared with SageAttention 2 |
| M4 | FP4 path, compared with SageAttention 3 |
| M5 | ComfyUI integration, real WAN 2.2 video |

## Results

Every result goes to [rigsy.dev/bench](https://rigsy.dev/bench), wins and losses alike.
