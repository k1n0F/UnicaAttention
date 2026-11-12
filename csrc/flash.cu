/*
 * UnicaAttention's production kernels: attention_int8pv_tma (the accurate mode: INT8 Q·Kᵀ, u8 × s8 P·V) and
 * attention_fp4 (the fast mode: NVFP4), and the quantizers they read from.
 *
 * Everything that led to them (the step kernels of M1–M4, every build setting that measured a change, the TRACE
 * timeline) is in legacy.cu, frozen as it was when the two were split (audit, after M4.55). unica.attention builds
 * legacy.cu for a legacy kernel or setting, so every measurement in LOG.md can still be run again.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cstdint>

constexpr int MMA_N = 8;  // the N of every mma here: D comes in blocks of 16 × 8

// The build settings production has: the two layouts of the accurate kernel (4 computing warps, or 8 with the
// producer: M4.48, M4.53), its V scale (per channel, per token M4.23, or per channel without outliers M4.25), and the
// fast kernel's Q smoothing (M4.21). unica's presets and wave_layout choose among them.
#ifndef UNICA_WARPS_PER_BLOCK
#define UNICA_WARPS_PER_BLOCK 4
#endif
// M4.23: 1 gives attention_int8pv_tma V with one scale per token, folded into P (online_softmax_u8_token); the token
// scales are copied into shared memory with V (M4.41).
#ifndef UNICA_V_TOKEN
#define UNICA_V_TOKEN 0
#endif
// M4.48: 1 gives attention_int8pv_tma a producer warpgroup: 4 more warps, which drop to 24 registers (setmaxnreg) while
// the computing warps rise; one producer thread issues every copy, and the computing warps never wait for each other
// (no __syncthreads per tile). The host launches 4 more warps.
#ifndef UNICA_PRODUCER
#define UNICA_PRODUCER 0
#endif

constexpr int WARP_SIZE = 32;
constexpr int HEAD_DIM = 128;      // WAN 2.2
constexpr int ROWS_PER_WARP = 16;  // query rows: the M of one mma
constexpr int KEYS_PER_TILE = 64;  // keys one warp scores per pass

// acc += A · Bᵀ on one warp. A is 16 × MMA_K, B is MMA_N × MMA_K, both row-major bf16.
// B row-major is exactly the "col" operand PTX wants, so K needs no transpose for Q·Kᵀ.
//
// Each lane holds a piece of every matrix. With row = lane / 4 and col = 2 · (lane % 4):
//   a[0] = A[row][col, col+1]       a[1] = A[row+8][col, col+1]
//   a[2] = A[row][col+8, col+9]     a[3] = A[row+8][col+8, col+9]
//   b[0] = B[row][col, col+1]       b[1] = B[row][col+8, col+9]
//   acc[0..1] = D[row][col, col+1]  acc[2..3] = D[row+8][col, col+1]
// One uint32_t carries two neighbouring bf16 of one row.
__device__ __forceinline__ void mma_bf16_16816(float acc[4], const uint32_t a[4], const uint32_t b[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// M3: the same warp-wide mma for int8, m16n8k32: twice the k per instruction, int32 accumulate, exact.
// In bytes, the fragments match the bf16 m16n8k16 ones above: a row of 32 int8 is 32 bytes, like a row
// of 16 bf16, and each uint32_t carries 4 neighbouring int8 instead of 2 bf16:
//   a[0] = A[row][2·col .. 2·col+3]     a[1] = A[row+8][same]
//   a[2] = A[row][2·col+16 .. +19]      a[3] = A[row+8][same]
//   b[0] = B[row][2·col .. 2·col+3]     b[1] = B[row][2·col+16 .. +19]
//   acc as in fp32: acc[0..1] = D[row][col, col+1], acc[2..3] = D[row+8][col, col+1]
__device__ __forceinline__ void mma_s8_16832(int acc[4], const uint32_t a[4], const uint32_t b[2]) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

constexpr int KEY_BLOCKS = KEYS_PER_TILE / MMA_N;  // 8 blocks of 16 × 8 in one score tile

// M4.49: threadIdx.x read where it is used. asm volatile: ptxas can neither hoist the read out of its branch nor
// merge it with an earlier one, so values derived from it are computed there instead of living across a loop.
__device__ __forceinline__ int fresh_thread_index() {
    int index;
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(index));
    return index;
}

// Keys at or past n_keys score -inf, so p = 2^(-inf) = 0. In D, the lane's col picks the key.
__device__ __forceinline__ void mask_scores(float scores[KEY_BLOCKS][4], int col, int n_keys) {
#pragma unroll
    for (int block = 0; block < KEY_BLOCKS; ++block)
#pragma unroll
        for (int i = 0; i < 4; ++i)
            if (block * MMA_N + col + i % 2 >= n_keys) scores[block][i] = -CUDART_INF_F;
}

// A row of D is spread over 4 neighbouring lanes (4·row .. 4·row + 3). Two butterfly steps give
// every one of them the result: lane ^ 1 swaps with its neighbour, lane ^ 2 with the next pair.
__device__ __forceinline__ float quad_max(float x) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 1));
    return fmaxf(x, __shfl_xor_sync(0xffffffff, x, 2));
}

__device__ __forceinline__ float quad_sum(float x) {
    x += __shfl_xor_sync(0xffffffff, x, 1);
    return x + __shfl_xor_sync(0xffffffff, x, 2);
}

constexpr int D_BLOCKS = HEAD_DIM / MMA_N;        // 16 blocks of 16 × 8 across one row of O

__device__ __forceinline__ uint32_t pack_f16(float lo, float hi) {
    const __half2 pair = __floats2half2_rn(lo, hi);  // lo goes to the low 16 bits, the lower address
    return *reinterpret_cast<const uint32_t*>(&pair);
}

// What O has gathered so far was weighted against the old max: shrink it by alpha, per row.
__device__ __forceinline__ void rescale_out(float out[D_BLOCKS][4], const float alpha[2]) {
#pragma unroll
    for (int block = 0; block < D_BLOCKS; ++block) {
        out[block][0] *= alpha[0];
        out[block][1] *= alpha[0];
        out[block][2] *= alpha[1];
        out[block][3] *= alpha[1];
    }
}

// O = out / sum as bf16, two neighbouring values per 32-bit store. Rows at or past n_rows are not written.
// v_scale (M3.3): one scale per channel of a quantized V, applied here once instead of in every tile.
__device__ __forceinline__ void store_out(__nv_bfloat16* O, const float out[D_BLOCKS][4], const float row_sum[2],
                                          int row, int col, int n_rows, const float* v_scale = nullptr,
                                          long long row_stride = HEAD_DIM) {
    const float inv_sum[2] = {1.0f / quad_sum(row_sum[0]), 1.0f / quad_sum(row_sum[1])};
#pragma unroll
    for (int block = 0; block < D_BLOCKS; ++block) {
        const int dim = block * MMA_N + col;
        const float scale[2] = {v_scale ? v_scale[dim] : 1.0f, v_scale ? v_scale[dim + 1] : 1.0f};
        if (row < n_rows)
            *reinterpret_cast<__nv_bfloat162*>(O + row * row_stride + dim) = __floats2bfloat162_rn(
                out[block][0] * inv_sum[0] * scale[0], out[block][1] * inv_sum[0] * scale[1]);
        if (row + 8 < n_rows)
            *reinterpret_cast<__nv_bfloat162*>(O + (row + 8) * row_stride + dim) = __floats2bfloat162_rn(
                out[block][2] * inv_sum[1] * scale[0], out[block][3] * inv_sum[1] * scale[1]);
    }
}

constexpr int WARPS_PER_BLOCK = UNICA_WARPS_PER_BLOCK;
constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * ROWS_PER_WARP;  // 64 query rows per block

// ---------------------------------------------------------------------------------------------------
// M2: speed. Each step is a separate kernel next to the one before it, so bench/compare.py can time
// both in one run.
// ---------------------------------------------------------------------------------------------------

constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
constexpr int CHUNK = 8;                          // bf16 in one 16-byte load
constexpr int CHUNKS_PER_ROW = HEAD_DIM / CHUNK;  // 16

__device__ __forceinline__ uint32_t smem_address(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Four 8 × 8 bf16 matrices from shared memory in one instruction. Lanes 8m .. 8m + 7 point at the 8 rows
// (16 bytes each) of matrix m, and r[m] comes back in the mma layout: row lane / 4, columns
// 2 · (lane % 4) and + 1. volatile: it reads shared memory, so it must not move across __syncthreads.
__device__ __forceinline__ void ldmatrix_x4(uint32_t r[4], const void* lane_row) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_address(lane_row)));
}

// ---------------------------------------------------------------------------------------------------
// M3: low precision.
// ---------------------------------------------------------------------------------------------------

constexpr int MMA_K8 = 32;                    // int8 m16n8k32
constexpr int Q8_STEPS = HEAD_DIM / MMA_K8;  // 4 fragments of Q along d: 16 registers instead of 32
constexpr int Q_SCALE_BLOCK = 128;            // query tokens per Q scale
constexpr int K_SCALE_BLOCK = 64;             // keys per K scale
static_assert(Q_SCALE_BLOCK % ROWS_PER_BLOCK == 0, "a warp's rows must share one Q scale");
static_assert(K_SCALE_BLOCK % KEYS_PER_TILE == 0, "a tile's keys must share one K scale");

// A row of 128 int8 is 128 bytes, one full round of the 32 banks: pad by 16 bytes, so row i starts in
// bank 4·i, as with SMEM_STRIDE.
constexpr int K8_STRIDE = HEAD_DIM + 16;

// Where key k of a group of 16 sits in the V tile of attention_int8pv. P leaves Q·Kᵀ with keys 2t, 2t+1,
// 8+2t, 9+2t in lane t's four bytes (t = lane % 4), and m16n8k32 reads those bytes as k = 4t .. 4t+3. The
// sum over k does not care about the order, only that P and V agree: V's rows are stored in P's order,
// slot 4t + j holds key 2t + j % 2 + 8 · (j / 2). Keys 0, 1, 8, 9, 2, 3, 10, 11 .. go to slots 0, 1, 2, 3 ..
__device__ __forceinline__ int p_order_slot(int key) {
    const int k = key % 16;
    return key - k + 2 * (k % 8) - k % 2 + 2 * (k / 8);
}

// Integer scores q̂·k̂ for one tile, exact in int32, returned as float. ldmatrix moves bytes, so the bf16
// scheme carries over: one ldmatrix.x4 brings two key blocks of one 32-wide step of d (matrices 0, 1:
// keys 0..7, bytes 0..15 and 16..31; matrices 2, 3: keys 8..15). 16 ldmatrix and 32 mma per tile,
// half of qk_scores_smem.
__device__ __forceinline__ void qk_scores_int8(float scores[KEY_BLOCKS][4], const uint32_t q[Q8_STEPS][4],
                                               const int8_t* k_tile, int lane) {
    const int matrix = lane / 8, matrix_row = lane % 8;
    const int8_t* lane_row = k_tile + (matrix / 2 * MMA_N + matrix_row) * K8_STRIDE + matrix % 2 * 16;
    int acc[KEY_BLOCKS][4] = {};
#pragma unroll
    for (int step = 0; step < Q8_STEPS; ++step)
#pragma unroll
        for (int pair = 0; pair < KEY_BLOCKS / 2; ++pair) {
            uint32_t b[4];
            ldmatrix_x4(b, lane_row + pair * 2 * MMA_N * K8_STRIDE + step * MMA_K8);
            mma_s8_16832(acc[2 * pair], q[step], b);
            mma_s8_16832(acc[2 * pair + 1], q[step], b + 2);
        }
#pragma unroll
    for (int block = 0; block < KEY_BLOCKS; ++block)
#pragma unroll
        for (int i = 0; i < 4; ++i) scores[block][i] = __int2float_rn(acc[block][i]);
}

constexpr int QUANTIZE_THREADS = 256;

// Max over the whole block of threads; every thread gets it.
__device__ __forceinline__ float block_max(float x) {
    __shared__ float warp_max[QUANTIZE_THREADS / WARP_SIZE];
#pragma unroll
    for (int lanes = WARP_SIZE / 2; lanes > 0; lanes /= 2) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, lanes));
    if (threadIdx.x % WARP_SIZE == 0) warp_max[threadIdx.x / WARP_SIZE] = x;
    __syncthreads();
    x = warp_max[0];
#pragma unroll
    for (int warp = 1; warp < QUANTIZE_THREADS / WARP_SIZE; ++warp) x = fmaxf(x, warp_max[warp]);
    return x;
}

// 8 neighbouring values of one row as float, minus the mean of their dims if there is one.
// M4.17: where a token of one head starts in a q, k, v or output tensor whose last dimension is contiguous: [b, h, s, d]
// (contiguous) or [b, s, h, d] (ComfyUI's), or any other strides. head is batch · heads + head. Strides in elements.
// Passed by value; unica.TokenLayout must match it field for field (32 bytes).
struct TokenLayout {
    long long batch, head, token;
    int heads;
    __device__ __forceinline__ size_t offset(int head_index, int token_index) const {
        return size_t(head_index / heads) * batch + size_t(head_index % heads) * head + size_t(token_index) * token;
    }
};

__device__ __forceinline__ void load_chunk(float values[CHUNK], const __nv_bfloat16* src, const float* mean) {
    const uint4 raw = *reinterpret_cast<const uint4*>(src);
    const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int i = 0; i < CHUNK / 2; ++i) {
        const float2 pair = __bfloat1622float2(pairs[i]);
        values[2 * i] = pair.x - (mean ? mean[2 * i] : 0.0f);
        values[2 * i + 1] = pair.y - (mean ? mean[2 * i + 1] : 0.0f);
    }
}

// M4.18–M4.19: RoPE inside the quantization. freqs: (c, s) per pair of channels in fp16, [batch or 1][n_tokens]
// [HEAD_DIM / 2][2] (unica.rope_table compacts ComfyUI's float [[c, −s], [s, c]] table to it: M4.18 read the float
// table, 8 bytes per element of q, once per head, and lost to L2). out[2i] = c · x[2i] − s · x[2i + 1], out[2i + 1] =
// s · x[2i] + c · x[2i + 1], in float: each product rounded, then the difference or sum (no FMA contraction), as
// bench/emulate.py's rope() does it. freqs_batch: the table's batch stride in halves, 0 when one table serves every batch.
constexpr int ROPE_HALVES_PER_TOKEN = HEAD_DIM;

// The chunk of 8 channels at `dim` of one token, rotated; f: the table of that token (one 16-byte load).
__device__ __forceinline__ void rope_chunk(float values[CHUNK], const __half* f, int dim) {
    const uint4 raw = *reinterpret_cast<const uint4*>(f + dim);
    const __half2* cs = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
    for (int j = 0; j < CHUNK / 2; ++j) {
        const float2 rotation = __half22float2(cs[j]);  // (c, s)
        const float x0 = values[2 * j], x1 = values[2 * j + 1];
        values[2 * j] = __fsub_rn(__fmul_rn(rotation.x, x0), __fmul_rn(rotation.y, x1));
        values[2 * j + 1] = __fadd_rn(__fmul_rn(rotation.y, x0), __fmul_rn(rotation.x, x1));
    }
}

// load_chunk with RoPE first (f: the token's table, or null), then the mean subtracted (the mean of the rotated x)
__device__ __forceinline__ void load_chunk_rope(float values[CHUNK], const __nv_bfloat16* src, const float* mean,
                                                const __half* f, int dim) {
    load_chunk(values, src, nullptr);
    if (f) rope_chunk(values, f, dim);
    if (mean) {
#pragma unroll
        for (int i = 0; i < CHUNK; ++i) values[i] -= mean[i];
    }
}

// Per channel of a head: Σ, min and max of x after RoPE over a block of STATS_TOKENS tokens. One pass gives the K mean
// and the per-head max of the FP4 scale, on the rotated values, which exist nowhere in memory. partial: float
// [heads][blocks][3][HEAD_DIM] (Σ, min, max); unica.rope_stats adds the blocks up in a fixed order. Grid: (blocks,
// heads). Block: QUANTIZE_THREADS = 16 chunks of 8 channels × 16 tokens at a time.
constexpr int STATS_TOKENS = 256;
constexpr int STATS_LANES = QUANTIZE_THREADS / CHUNKS_PER_ROW;  // 16
extern "C" __global__ void __launch_bounds__(QUANTIZE_THREADS)
    rope_stats(const __nv_bfloat16* X, const __half* freqs, long long freqs_batch, float* partial, int n_tokens,
               TokenLayout in) {
    const int chunk = threadIdx.x % CHUNKS_PER_ROW, lane = threadIdx.x / CHUNKS_PER_ROW;
    const int head = blockIdx.y, token0 = blockIdx.x * STATS_TOKENS;
    const __half* table = freqs ? freqs + size_t(head / in.heads) * freqs_batch : nullptr;  // null: x as it is (M4.21)
    float sum[CHUNK], low[CHUNK], high[CHUNK];
#pragma unroll
    for (int i = 0; i < CHUNK; ++i) sum[i] = 0.0f, low[i] = CUDART_INF_F, high[i] = -CUDART_INF_F;
    for (int token = token0 + lane; token < min(token0 + STATS_TOKENS, n_tokens); token += STATS_LANES) {
        float values[CHUNK];
        load_chunk_rope(values, X + in.offset(head, token) + chunk * CHUNK, nullptr,
                        table ? table + size_t(token) * ROPE_HALVES_PER_TOKEN : nullptr, chunk * CHUNK);
#pragma unroll
        for (int i = 0; i < CHUNK; ++i) {
            sum[i] += values[i];
            low[i] = fminf(low[i], values[i]);
            high[i] = fmaxf(high[i], values[i]);
        }
    }
    __shared__ float stats[3][STATS_LANES][HEAD_DIM];
#pragma unroll
    for (int i = 0; i < CHUNK; ++i) {
        stats[0][lane][chunk * CHUNK + i] = sum[i];
        stats[1][lane][chunk * CHUNK + i] = low[i];
        stats[2][lane][chunk * CHUNK + i] = high[i];
    }
    __syncthreads();
    if (threadIdx.x < HEAD_DIM) {
        float s = 0.0f, lo = CUDART_INF_F, hi = -CUDART_INF_F;
        for (int l = 0; l < STATS_LANES; ++l) {
            s += stats[0][l][threadIdx.x];
            lo = fminf(lo, stats[1][l][threadIdx.x]);
            hi = fmaxf(hi, stats[2][l][threadIdx.x]);
        }
        float* out = partial + (size_t(head) * gridDim.x + blockIdx.x) * 3 * HEAD_DIM + threadIdx.x;
        out[0] = s;
        out[HEAD_DIM] = lo;
        out[2 * HEAD_DIM] = hi;
    }
}

// Quantization for attention_int8qk. One block of threads per scale block: block_tokens tokens of one head,
// x ≈ x8 · scale with x = X − mean (mean [heads][HEAD_DIM], or null). Bit for bit what bench/emulate.py's
// quant() computes: the same max, the same IEEE division (nvcc's default), round half to even like
// torch.round. The block is read twice; the second read comes from L2, not from VRAM.
// Grid: (ceil(n_tokens / block_tokens), heads). Block: QUANTIZE_THREADS. X in any TokenLayout; X8 [heads][n_tokens][HEAD_DIM].
extern "C" __global__ void __launch_bounds__(QUANTIZE_THREADS)
    quantize_int8(const __nv_bfloat16* X, const float* mean, int8_t* X8, float* scale, int n_tokens, int block_tokens,
                  TokenLayout in, const __half* freqs, long long freqs_batch) {
    const int token0 = blockIdx.x * block_tokens;
    const int n_chunks = min(block_tokens, n_tokens - token0) * CHUNKS_PER_ROW;
    X8 += (size_t(blockIdx.y) * n_tokens + token0) * HEAD_DIM;
    // chunk c of the block: token token0 + c / CHUNKS_PER_ROW, dims c % CHUNKS_PER_ROW · CHUNK (X in its own layout)
    auto x_chunk = [&](int chunk) { return X + in.offset(blockIdx.y, token0 + chunk / CHUNKS_PER_ROW) + chunk % CHUNKS_PER_ROW * CHUNK; };
    // M4.18: freqs (or null): RoPE on the way in, then the mean of the rotated x
    const __half* table = freqs ? freqs + size_t(blockIdx.y / in.heads) * freqs_batch : nullptr;
    auto rope_of = [&](int chunk) { return table ? table + size_t(token0 + chunk / CHUNKS_PER_ROW) * ROPE_HALVES_PER_TOKEN : nullptr; };
    if (mean) mean += blockIdx.y * HEAD_DIM;

    float local_max = 0.0f;
    for (int chunk = threadIdx.x; chunk < n_chunks; chunk += QUANTIZE_THREADS) {
        const int dim = chunk % CHUNKS_PER_ROW * CHUNK;
        float values[CHUNK];
        load_chunk_rope(values, x_chunk(chunk), mean ? mean + dim : nullptr, rope_of(chunk), dim);
#pragma unroll
        for (int i = 0; i < CHUNK; ++i) local_max = fmaxf(local_max, fabsf(values[i]));
    }
    // · (1/127), not / 127: torch divides a tensor by a scalar as a multiplication by its reciprocal, and
    // the two differ in the last bit for some maxima. The emulator is the reference, so match it.
    const float block_scale = fmaxf(block_max(local_max) * (1.0f / 127.0f), 1e-12f);
    if (threadIdx.x == 0) scale[size_t(blockIdx.y) * gridDim.x + blockIdx.x] = block_scale;

    for (int chunk = threadIdx.x; chunk < n_chunks; chunk += QUANTIZE_THREADS) {
        const int dim = chunk % CHUNKS_PER_ROW * CHUNK;
        float values[CHUNK];
        load_chunk_rope(values, x_chunk(chunk), mean ? mean + dim : nullptr, rope_of(chunk), dim);
        uint32_t packed[2] = {0, 0};
#pragma unroll
        for (int i = 0; i < CHUNK; ++i) {
            const int value = __float2int_rn(fminf(fmaxf(rintf(values[i] / block_scale), -127.0f), 127.0f));
            packed[i / 4] |= uint32_t(value & 0xff) << (8 * (i % 4));
        }
        *reinterpret_cast<uint2*>(X8 + chunk * CHUNK) = make_uint2(packed[0], packed[1]);
    }
}

// ---------------------------------------------------------------------------------------------------
// M3.3: P·V in 8 bits too, P as u8 and V as s8 (the u8 × s8 m16n8k32 runs at the s8 × s8 rate).
// ---------------------------------------------------------------------------------------------------

constexpr int PV8_STEPS = KEYS_PER_TILE / MMA_K8;  // 2 steps of 32 keys in P·V
constexpr float LOG2_255 = 7.99435343685886f;      // P comes out of the exponent already times 255

// out += A · Bᵀ with A u8 (16 × 32) and B s8 (8 × 32), int32 accumulate: the layout of mma_s8_16832.
__device__ __forceinline__ void mma_u8s8_16832(int acc[4], const uint32_t a[4], const uint32_t b[2]) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Two 16 × 16 int8 matrices from shared memory, transposed on the way in (8-bit ldmatrix, new in sm_120a).
// Lanes 0..15 point at the 16 rows of matrix 0, lanes 16..31 at those of matrix 1. Measured on the card: the
// lane gets rows 4·(lane % 4) .. + 3 of column lane / 4 in r[0] (matrix 0) and r[2] (matrix 1), and of
// column lane / 4 + 8 in r[1] and r[3]. With rows = keys and columns = dims, {r[0], r[2]} and {r[1], r[3]}
// are the B fragments of m16n8k32 for two neighbouring blocks of 8 dims.
__device__ __forceinline__ void ldmatrix_x2_trans_b8(uint32_t r[4], const void* lane_row) {
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_address(lane_row)));
}

// 2^x as the bare SFU instruction. exp2f is the same MUFU.EX2 inside a range check (FSETP, two predicated
// FMULs) that keeps results below 2^-126 as subnormals; here they become 0. Equal to exp2f bit for bit for
// every x ≥ −126 (legacy.cu's exp2_ftz_mismatches checks all of them).
__device__ __forceinline__ float exp2_ftz(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// online_softmax for a u8 P. Each row's P is taken against the max of this tile, not the running max, and
// comes out as 255 · 2^(s · scale_log2 − tile max), in [0, 255]: the largest p of every row and tile is 255,
// so a tile under an older, larger max still gets all 256 steps. beta = 2^(tile max − running max) brings it
// back when P·V is added into out. row_sum gathers the unrounded values: it is 255 times the usual sum, P·V
// carries the same 255, and the two cancel in out / sum. row_sum is left alone here: add_row_sum_u8 adds the
// rounded P̂ after packing (M3.15).
__device__ __forceinline__ void online_softmax_u8(float scores[KEY_BLOCKS][4], float row_max[2], float alpha[2],
                                                  float beta[2], float scale_log2) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        float tile_max = -CUDART_INF_F;
#pragma unroll
        for (int block = 0; block < KEY_BLOCKS; ++block)
            tile_max = fmaxf(tile_max, fmaxf(scores[block][2 * half], scores[block][2 * half + 1]));
        tile_max = quad_max(tile_max) * scale_log2;  // every tile has a valid key, so this is finite
        const float new_max = fmaxf(row_max[half], tile_max);
        alpha[half] = exp2_ftz(row_max[half] - new_max);
        beta[half] = exp2_ftz(tile_max - new_max);

        const float offset = LOG2_255 - tile_max;
#pragma unroll
        for (int block = 0; block < KEY_BLOCKS; ++block)
#pragma unroll
            for (int i = 2 * half; i < 2 * half + 2; ++i) scores[block][i] = exp2_ftz(fmaf(scores[block][i], scale_log2, offset));
        row_max[half] = new_max;
    }
}

// M3.15: row_sum = row_sum · alpha + (Σ P̂) · beta over the rounded u8 P̂ the mma uses, after packing. IDP4A adds
// four bytes at once, exact in int32: registers 0 and 2 of a step hold this lane's part of row r, 1 and 3 of
// row r + 8. The float P are then dead once packed, where the FADD of their sum kept 32 of them alive into
// P·V (M3.14). Σ P̂ instead of Σ P: the same on random data, up to +0.05 % error on sharp attention (LOG.md).
__device__ __forceinline__ void add_row_sum_u8(float row_sum[2], const uint32_t p[PV8_STEPS][4], const float alpha[2],
                                               const float beta[2]) {
    constexpr uint32_t ONES = 0x01010101;
    unsigned sum[2] = {0, 0};
#pragma unroll
    for (int step = 0; step < PV8_STEPS; ++step)
#pragma unroll
        for (int i = 0; i < 4; ++i) sum[i % 2] = __dp4a(p[step][i], ONES, sum[i % 2]);
#pragma unroll
    for (int half = 0; half < 2; ++half)
        row_sum[half] = row_sum[half] * alpha[half] + __uint2float_rn(sum[half]) * beta[half];
}

// M4.23: online_softmax_u8 with V scaled per token (quantize_v's v_token). tok[STRIDE · b]: this lane's two keys of
// block b, {log2 s, log2 s, 1 / s, 1 / s} of keys 8b + col and + 1 (STRIDE 1: registers; MMA_N / 2: a tile in shared
// memory, M4.41). A score becomes x = s · scale_log2 + log2 s_j, and
// 255 · 2^(x − tile max) is P' = p · s_j against the max of P' in the tile, so every key's V row keeps its own
// 127 steps, outlier tokens or not. The denominator is Σ p = Σ P' / s_j, gathered here in float before rounding
// (8-bit weights cannot span the s_j of outliers, ~250× apart in Z-Image); add_row_sum_u8 is not used.
template <int STRIDE>
__device__ __forceinline__ void online_softmax_u8_token(float scores[KEY_BLOCKS][4], float row_max[2], float row_sum[2],
                                                        float alpha[2], float beta[2], float scale_log2,
                                                        const float4* tok) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        float tile_max = -CUDART_INF_F;
#pragma unroll
        for (int block = 0; block < KEY_BLOCKS; ++block) {
            float& x0 = scores[block][2 * half];
            float& x1 = scores[block][2 * half + 1];
            x0 = fmaf(x0, scale_log2, tok[STRIDE * block].x);
            x1 = fmaf(x1, scale_log2, tok[STRIDE * block].y);
            tile_max = fmaxf(tile_max, fmaxf(x0, x1));
        }
        tile_max = quad_max(tile_max);
        const float new_max = fmaxf(row_max[half], tile_max);
        alpha[half] = exp2_ftz(row_max[half] - new_max);
        beta[half] = exp2_ftz(tile_max - new_max);

        const float offset = LOG2_255 - tile_max;
        float tile_sum = 0.0f;
#pragma unroll
        for (int block = 0; block < KEY_BLOCKS; ++block) {
            float& p0 = scores[block][2 * half];
            float& p1 = scores[block][2 * half + 1];
            p0 = exp2_ftz(p0 + offset);
            p1 = exp2_ftz(p1 + offset);
            tile_sum = fmaf(p0, tok[STRIDE * block].z, fmaf(p1, tok[STRIDE * block].w, tile_sum));
        }
        row_sum[half] = row_sum[half] * alpha[half] + tile_sum * beta[half];
        row_max[half] = new_max;
    }
}

// Four values in [0, 255] rounded half to even (torch.round) into one register, v0 in the lowest byte.
// cvt.pack.sat.u8.s32.b32 d, a, b, c gives d = c << 16 | sat(a) << 8 | sat(b) (measured), so the pair packed
// first ends up on top.
__device__ __forceinline__ uint32_t pack_u8(float v0, float v1, float v2, float v3) {
    uint32_t high, word;
    asm("cvt.pack.sat.u8.s32.b32 %0, %1, %2, 0;" : "=r"(high) : "r"(__float2int_rn(v3)), "r"(__float2int_rn(v2)));
    asm("cvt.pack.sat.u8.s32.b32 %0, %1, %2, %3;"
        : "=r"(word) : "r"(__float2int_rn(v1)), "r"(__float2int_rn(v0)), "r"(high));
    return word;
}

// The A fragment of u8 P for one step of 32 keys: four 16 × 8 blocks of scores, keys in p_order_slot order.
__device__ __forceinline__ void p_fragment_u8(uint32_t p[4], const float scores[KEY_BLOCKS][4], int step) {
    const float* b0 = scores[4 * step];
    const float* b1 = scores[4 * step + 1];
    const float* b2 = scores[4 * step + 2];
    const float* b3 = scores[4 * step + 3];
    p[0] = pack_u8(b0[0], b0[1], b1[0], b1[1]);
    p[1] = pack_u8(b0[2], b0[3], b1[2], b1[3]);
    p[2] = pack_u8(b2[0], b2[1], b3[0], b3[1]);
    p[3] = pack_u8(b2[2], b2[3], b3[2], b3[3]);
}

// out = out · alpha + beta · (P̂ · V̂) for one tile, V̂ in shared memory in p_order_slot order. P̂ u8 × V̂ s8 is
// exact in int32, and |sum| ≤ 64 · 255 · 127 < 2^24 keeps even the conversion to float exact. One pair of d blocks
// takes both 32-key steps and is added into out at once, so 8 int32 accumulators are alive at a time, not 64. V̂ in
// the paired rows of v8_offset's layout 2 (M3.17), V8_STRIDE apart, whose ldmatrix returns each B operand in adjacent
// registers: no MOV to pair them.
constexpr int V8_STRIDE = 2 * HEAD_DIM + 16;  // a paired row and 16 bytes of padding: row i starts in bank 4 · i
static_assert(KEYS_PER_TILE / 2 * V8_STRIDE <= KEYS_PER_TILE * K8_STRIDE, "a paired V tile must fit a V tile");

__device__ __forceinline__ void pv_int8(float out[D_BLOCKS][4], const uint32_t p[PV8_STEPS][4], const int8_t* v_tile,
                                        const float alpha[2], const float beta[2], int lane) {
    // M3.9: alpha is exactly 1 unless the max of a row grew in this tile, and then out · alpha is out. When it
    // is 1 in every row of the warp, the 64 multiplies are skipped. Rounding is the same on both paths.
    if (!__all_sync(0xffffffff, alpha[0] == 1.0f && alpha[1] == 1.0f)) rescale_out(out, alpha);
    // row j of the step for both matrices, matrix 0 reading the left block of dims of the pair, matrix 1 the right one
    const int8_t* lane_row = v_tile + lane % 16 * V8_STRIDE + lane / 16 * 16;
#pragma unroll
    for (int pair = 0; pair < D_BLOCKS / 2; ++pair) {
        int acc[2][4] = {};
#pragma unroll
        for (int step = 0; step < PV8_STEPS; ++step) {
            uint32_t b[4];
            ldmatrix_x2_trans_b8(b, lane_row + step * 16 * V8_STRIDE + pair * 32);
            const uint32_t left[2] = {b[0], b[1]}, right[2] = {b[2], b[3]};
            mma_u8s8_16832(acc[0], p[step], left);
            mma_u8s8_16832(acc[1], p[step], right);
        }
#pragma unroll
        for (int half = 0; half < 2; ++half)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float& o = out[2 * pair + half][i];
                o = fmaf(beta[i / 2], __int2float_rn(acc[half][i]), o);
            }
    }
}

// M3.17: V8 in global memory as TMA copies it. TMA writes rows in order, so the rows are put in P's order
// (p_order_slot) here instead of in the loader, and each head is padded to whole tiles of V8_PAD_KEYS keys
// (their bytes are never read with a P̂ other than 0). Layout 1: one row of HEAD_DIM bytes per slot. Layout 2:
// the pairs of M3.11, so that ldmatrix returns the B operands of P·V in adjacent registers: a row of
// 2 · HEAD_DIM bytes per slot j of each 16 in a 32-key step, holding for every block of 8 dims those dims of
// slot j, then of slot j + 16. The byte offset of chunk `chunk` (dims 8c .. 8c + 7 of one key) of quantize_v.
constexpr int V8_PAD_KEYS = 64;

__host__ __device__ constexpr int v8_tokens(int n_tokens) {
    return (n_tokens + V8_PAD_KEYS - 1) / V8_PAD_KEYS * V8_PAD_KEYS;
}

__device__ __forceinline__ size_t v8_offset(int layout, int chunk, int n_tokens) {
    if (layout == 0) return size_t(chunk) * CHUNK;
    const int head = chunk / (n_tokens * CHUNKS_PER_ROW);
    const int key = chunk / CHUNKS_PER_ROW % n_tokens;
    const int c = chunk % CHUNKS_PER_ROW;
    const int slot = p_order_slot(key);
    const size_t head_base = size_t(head) * v8_tokens(n_tokens) * HEAD_DIM;
    if (layout == 1) return head_base + size_t(slot) * HEAD_DIM + c * CHUNK;
    return head_base + size_t(slot / MMA_K8 * 16 + slot % 16) * 2 * HEAD_DIM + c * 16 + slot % MMA_K8 / 16 * 8;
}

// V for attention_int8pv: v ≈ v8 · v_scale[dim], one scale per channel over all tokens of a head. d is the
// axis P·V does not sum over, so the scale stays out of the mma and store_out applies it once. v_scale comes
// from torch: the max of a channel needs every token. Bit for bit bench/emulate.py: IEEE division, round half
// to even, clamp to ±127. Grid: ceil(n_chunks / QUANTIZE_THREADS), n_chunks = heads · n_tokens · CHUNKS_PER_ROW.
// layout (M3.17, see v8_offset): 0 [heads][n_tokens][HEAD_DIM]; 1 and 2 for TMA, [heads][v8_tokens][HEAD_DIM]
// bytes each.
//
// v_token (M4.23, else null): V gets one scale per token instead, s_j = max |v_j| / 127 over its channels (v_scale is
// then not read), and v_token [heads][v8_tokens / 2] float4 gets, for every pair of keys 2m, 2m + 1,
// {log2 s_2m, log2 s_2m+1, 1 / s_2m, 1 / s_2m+1}: attention_int8pv_tma with UNICA_V_TOKEN adds log2 s_j to key j's
// scores, so P' = p · s_j goes into the mma, and weighs the denominator back by 1 / s_j. Bit for bit
// bench/emulate.py's v_token V. The 16 chunks of a token are 16 neighbouring lanes, whose max is one xor shuffle tree.
// skip (M4.25, else null): uint8 [heads][n_tokens], 1 for the keys written as zeros (V_OUTLIERS).
extern "C" __global__ void __launch_bounds__(QUANTIZE_THREADS)
    quantize_v(const __nv_bfloat16* V, const float* v_scale, int8_t* V8, int n_tokens, int n_chunks, int layout,
               TokenLayout in, float* v_token, const uint8_t* skip) {
    const int chunk = blockIdx.x * QUANTIZE_THREADS + threadIdx.x;
    if (chunk >= n_chunks) return;  // whole tokens: n_chunks is a multiple of CHUNKS_PER_ROW, the blocks too
    const int head = chunk / (n_tokens * CHUNKS_PER_ROW), key = chunk / CHUNKS_PER_ROW % n_tokens;
    float values[CHUNK];
    load_chunk(values, V + in.offset(head, key) + chunk % CHUNKS_PER_ROW * CHUNK, nullptr);
    if (skip && skip[size_t(head) * n_tokens + key]) {  // M4.25: an outlier key, added exactly after the loop
#pragma unroll
        for (int i = 0; i < CHUNK; ++i) values[i] = 0.0f;
    }
    float scale[CHUNK];
    if (v_token) {
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < CHUNK; ++i) amax = fmaxf(amax, fabsf(values[i]));
        const unsigned group = 0xffffu << (threadIdx.x % WARP_SIZE / CHUNKS_PER_ROW * CHUNKS_PER_ROW);
#pragma unroll
        for (int lanes = CHUNKS_PER_ROW / 2; lanes > 0; lanes /= 2)
            amax = fmaxf(amax, __shfl_xor_sync(group, amax, lanes));
        const float s = fmaxf(amax / 127.0f, 1e-12f);  // as torch: amax / 127, then clamp_min
#pragma unroll
        for (int i = 0; i < CHUNK; ++i) scale[i] = s;
        if (chunk % CHUNKS_PER_ROW == 0) {
            float* pair = v_token + (size_t(head) * v8_tokens(n_tokens) + key / 2 * 2) * 2;
            pair[key % 2] = log2f(s);
            pair[2 + key % 2] = 1.0f / s;
        }
    } else {
#pragma unroll
        for (int i = 0; i < CHUNK; ++i) scale[i] = v_scale[head * HEAD_DIM + chunk % CHUNKS_PER_ROW * CHUNK + i];
    }
    uint32_t packed[2] = {0, 0};
#pragma unroll
    for (int i = 0; i < CHUNK; ++i) {
        const int value = __float2int_rn(fminf(fmaxf(rintf(values[i] / scale[i]), -127.0f), 127.0f));
        packed[i / 4] |= uint32_t(value & 0xff) << (8 * (i % 4));
    }
    *reinterpret_cast<uint2*>(V8 + v8_offset(layout, chunk, n_tokens)) = make_uint2(packed[0], packed[1]);
}

// ---------------------------------------------------------------------------------------------------
// M3.4: tiles load while the previous one is computed.
// ---------------------------------------------------------------------------------------------------

// 16 bytes global -> shared without passing through registers: the thread does not wait for them, it waits
// later in cp_async_wait. src_bytes 0 reads nothing and writes 16 zeros. .cg: through L2 only, the tile is
// read once per block and would only push other data out of L1.
__device__ __forceinline__ void cp_async16(void* dst, const void* src, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(smem_address(dst)), "l"(src), "r"(src_bytes) : "memory");
}

// Every cp_async16 of this thread issued before it is one group; cp_async_wait waits for all groups.
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }

// ---------------------------------------------------------------------------------------------------
// M3.12: Q·Kᵀ of the next tile issued before the softmax of this one.
// ---------------------------------------------------------------------------------------------------

// M3.13: the warp's 16 rows of int8 Q into shared memory, K8_STRIDE apart, joining the caller's cp.async
// group. Rows at or past n_rows become zeros, read from nowhere: their address stays at Q8, row 0 of the head,
// which always exists.
__device__ __forceinline__ void load_q8_async(int8_t* q_tile, const int8_t* Q, const int8_t* Q8, int n_rows,
                                              int lane) {
#pragma unroll
    for (int load = 0; load < ROWS_PER_WARP * HEAD_DIM / 16 / WARP_SIZE; ++load) {
        const int chunk = load * WARP_SIZE + lane;
        const int q_row = chunk / (HEAD_DIM / 16), byte = chunk % (HEAD_DIM / 16) * 16;
        const bool valid = q_row < n_rows;
        cp_async16(q_tile + q_row * K8_STRIDE + byte, valid ? Q + q_row * HEAD_DIM + byte : Q8, valid ? 16 : 0);
    }
}

// qk_scores_int8 with Q from the warp's tile in shared memory (load_q8_async). One ldmatrix.x4 per 32-wide step
// of d returns the A fragment load_q8 builds (row lane / 4, bytes 4 · (lane % 4); matrices: rows 0–7 and
// 8–15 of bytes 0–15, then of bytes 16–31), so Q takes no registers between tiles.
__device__ __forceinline__ void qk_scores_int8_q_smem(float scores[KEY_BLOCKS][4], const int8_t* q_tile,
                                                      const int8_t* k_tile, int lane) {
    const int matrix = lane / 8;
    const int8_t* lane_row = q_tile + (lane % 8 + matrix % 2 * 8) * K8_STRIDE + matrix / 2 * 16;
    uint32_t q[Q8_STEPS][4];
#pragma unroll
    for (int step = 0; step < Q8_STEPS; ++step) ldmatrix_x4(q[step], lane_row + step * MMA_K8);
    qk_scores_int8(scores, q, k_tile, lane);
}

// t % 2 as a type, for the two halves of the loop in attention_int8pv_pipe.
template <int VALUE>
struct Parity {
    static constexpr int value = VALUE;
};

// ---------------------------------------------------------------------------------------------------
// M3.17: tiles by TMA.
// ---------------------------------------------------------------------------------------------------

// An mbarrier in shared memory that `count` arrivals complete, plus the bytes announced with expect_bytes.
__device__ __forceinline__ void mbarrier_init(uint64_t* barrier, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_address(barrier)), "r"(count));
}

// This thread's arrival, announcing `bytes` that TMA copies will deliver before the phase completes.
__device__ __forceinline__ void mbarrier_expect_bytes(uint64_t* barrier, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_address(barrier)), "r"(bytes)
                 : "memory");
}

// This thread's arrival alone; it releases the thread's earlier reads and writes to whoever waits on the phase.
__device__ __forceinline__ void mbarrier_arrive(uint64_t* barrier) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_address(barrier)) : "memory");
}

// Waits until the phase with parity `phase` of the barrier has completed. (A suspend-time hint changed nothing,
// M4.51: legacy.cu's WAIT_HINT.)
__device__ __forceinline__ void mbarrier_wait(uint64_t* barrier, int phase) {
    uint32_t done = 0;
    while (!done)
        asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                     : "=r"(done)
                     : "r"(smem_address(barrier)), "r"(phase)
                     : "memory");
}

// One box of a 2D tensor map, starting at element x of row y, into shared memory (128-byte aligned); its bytes
// count towards `barrier`. Elements outside the tensor arrive as zeros. M4.48: the destination is .shared::cta. With
// .shared::cluster ptxas checks whether the address is another block's and, if so, calls a driver routine
// (__cuda_syscall_cp_async_bulk_*): never taken here, but an extern call in a kernel makes ptxas ignore setmaxnreg.
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap& map, int x, int y, uint64_t* barrier) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        ::"r"(smem_address(dst)), "l"(&map), "r"(x), "r"(y), "r"(smem_address(barrier))
        : "memory");
}

// `bytes` (a multiple of 16) from global memory into shared memory (both 16-byte aligned), counted on `barrier`
__device__ __forceinline__ void bulk_load(void* dst, const void* src, int bytes, uint64_t* barrier) {
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(smem_address(dst)), "l"(src), "r"(bytes), "r"(smem_address(barrier)) : "memory");
}

constexpr int K_TILE_BYTES = KEYS_PER_TILE * K8_STRIDE;  // a box of 144 bytes × KEYS_PER_TILE rows
constexpr bool TOKEN_TILES = UNICA_V_TOKEN;  // M4.41: copied with V
constexpr int TOK_TILE_BYTES = KEYS_PER_TILE / 2 * sizeof(float4);  // a tile's v_token, one float4 per pair of keys
constexpr int V_TILE_BYTES = KEYS_PER_TILE / 2 * V8_STRIDE;

// M4.25: the part of P·V of a head's V_OUTLIERS keys with the largest max |v|, which quantize_v left out of V8 (zero
// rows; the channel scales come from the other keys). In the loop their P̂ went into the denominator as any key's; here,
// once, their P against the final row max is added into out, exactly: P = 255 · 2^(x − m) with x their INT8 score as
// the loop computes it, times their V row over the channel scale (out_v, float). out_idx: [heads][V_OUTLIERS] keys
// (−1: none); out_v: [heads][V_OUTLIERS][HEAD_DIM]. Lane l scores row l % 16 against keys l / 16, + 2, ... with IDP4A
// on Q8 in shared memory and K8 in global memory; the out update takes each row's P from there by shuffles.
// In Z-Image two tokens (positions 1 and 2 of 4128, likely attention sinks) set every channel's scale (M4.24).
#ifndef UNICA_V_OUTLIERS
#define UNICA_V_OUTLIERS 0
#endif
__device__ __forceinline__ void add_v_outliers(float out[D_BLOCKS][4], const float row_max[2], const int8_t* q_tile,
                                               const int8_t* K8, const int* out_idx, const float* out_v,
                                               const float* k_scale, float q_scale_log2, int n_keys, int lane) {
    constexpr int T = UNICA_V_OUTLIERS > 0 ? UNICA_V_OUTLIERS : 1;
    constexpr int MINE = (T + 1) / 2;
    const int row = lane % 16;
    // the final max of `row`: lane 4 · (row % 8) holds it, in row_max[0] for rows 0..7 and row_max[1] for 8..15
    const float max0 = __shfl_sync(0xffffffff, row_max[0], 4 * (row % 8));
    const float max1 = __shfl_sync(0xffffffff, row_max[1], 4 * (row % 8));
    const float m = row < 8 ? max0 : max1;
    const int* keys = out_idx + size_t(blockIdx.y) * T;
    float p[MINE];
#pragma unroll
    for (int i = 0; i < MINE; ++i) {
        const int t = lane / 16 + 2 * i;
        const int key = t < T ? keys[t] : -1;
        p[i] = 0.0f;
        if (key >= 0) {
            const int* q_row = reinterpret_cast<const int*>(q_tile + row * K8_STRIDE);
            const int* k_row = reinterpret_cast<const int*>(K8 + (size_t(blockIdx.y) * n_keys + key) * HEAD_DIM);
            int dot = 0;
#pragma unroll 8
            for (int c = 0; c < HEAD_DIM / 4; ++c) dot = __dp4a(q_row[c], __ldg(k_row + c), dot);
            const float x = __int2float_rn(dot) * q_scale_log2 * k_scale[key / K_SCALE_BLOCK];
            p[i] = exp2f(x - m + LOG2_255);
        }
    }
    const int g = lane / 4, col = 2 * (lane % 4);
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const float p_g = __shfl_sync(0xffffffff, p[t / 2], g + 16 * (t % 2));
        const float p_g8 = __shfl_sync(0xffffffff, p[t / 2], g + 8 + 16 * (t % 2));
        const float* v_row = out_v + (size_t(blockIdx.y) * T + t) * HEAD_DIM;
#pragma unroll
        for (int block = 0; block < D_BLOCKS; ++block) {
            const float2 w = __ldg(reinterpret_cast<const float2*>(v_row + block * MMA_N + col));
            out[block][0] = fmaf(p_g, w.x, out[block][0]);
            out[block][1] = fmaf(p_g, w.y, out[block][1]);
            out[block][2] = fmaf(p_g8, w.x, out[block][2]);
            out[block][3] = fmaf(p_g8, w.y, out[block][3]);
        }
    }
}

// M3.17: attention_int8pv_pipe with its tiles copied by TMA. One thread issues each tile and announces its
// bytes on an mbarrier; every thread waits on the barrier, none computes an address. The boxes are 16 bytes
// wider than a row of K8 or V8, so the tiles land with the padded stride the ldmatrix addressing expects.
// k_map: K8 as rows of HEAD_DIM bytes, box K8_STRIDE × KEYS_PER_TILE. v_map: V8 in quantize_v layout 2 (uint16 rows
// of 2 · HEAD_DIM bytes, box V8_STRIDE bytes × KEYS_PER_TILE / 2). Q, scales, grid and block as attention_int8pv. The maps come first: TMA needs them 64-byte aligned,
// and a map after the 8-byte Q8 lands at offset 8 of the parameters (cudaErrorMisalignedAddress, M3.17).
// M4.48: with UNICA_PRODUCER the producer warpgroup's 128 threads come after the computing warps. 4 computing warps:
// the block launches at 128 registers per thread (2 blocks per SM), the computing warps take 232. M4.53, 8 computing
// warps (two warpgroups, 128 query rows sharing every K and V tile): 1 block per SM at 168 per thread, and the
// producer's (168 − 24) · 128 = 18,432 registers give the 256 computing threads 240 each.
constexpr int PRODUCER_THREADS = UNICA_PRODUCER ? 128 : 0;
constexpr int COMPUTING_REGISTERS = WARPS_PER_BLOCK == 8 ? 240 : 232;
static_assert(!UNICA_PRODUCER || WARPS_PER_BLOCK == 4 || WARPS_PER_BLOCK == 8,
              "setmaxnreg works on whole warpgroups: 4 or 8 computing warps");
extern "C" __global__ void __launch_bounds__(THREADS_PER_BLOCK + PRODUCER_THREADS, UNICA_PRODUCER ? 8 / WARPS_PER_BLOCK : 1)
    attention_int8pv_tma(const __grid_constant__ CUtensorMap k_map, const __grid_constant__ CUtensorMap v_map,
                         const int8_t* Q8, const float* q_scale, const float* k_scale, const float* v_scale,
                         const float4* v_token, const int8_t* K8, const int* out_idx, const float* out_v,
                         __nv_bfloat16* O, int n_queries, int n_keys, float scale_log2, TokenLayout o_layout) {
    __shared__ __align__(128) int8_t k_tiles[2][K_TILE_BYTES];
    __shared__ __align__(128) int8_t v_tiles[2][KEYS_PER_TILE * K8_STRIDE];
    __shared__ __align__(16) int8_t q_tiles[WARPS_PER_BLOCK][ROWS_PER_WARP * K8_STRIDE];
    // M4.41: tok_tiles[b] holds the token scales of the tile whose V is in v_tiles[b] (one float4 when unused)
    __shared__ __align__(16) float4 tok_tiles[2][TOKEN_TILES ? KEYS_PER_TILE / 2 : 1];
    // landed[p]: the copies issued in iterations t with t % 2 == p; landed[2]: those of the prologue
    __shared__ __align__(8) uint64_t landed[3];

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int col = 2 * (lane % 4);

    const int row0 = blockIdx.x * ROWS_PER_BLOCK + warp * ROWS_PER_WARP;
    const int n_rows = min(n_queries - row0, ROWS_PER_WARP);

    const size_t q_offset = (size_t(blockIdx.y) * n_queries + row0) * HEAD_DIM;
    v_scale += size_t(blockIdx.y) * HEAD_DIM;
    k_scale += size_t(blockIdx.y) * ((n_keys + K_SCALE_BLOCK - 1) / K_SCALE_BLOCK);
    const float q_scale_log2 =
        q_scale[size_t(blockIdx.y) * ((n_queries + Q_SCALE_BLOCK - 1) / Q_SCALE_BLOCK) + row0 / Q_SCALE_BLOCK] *
        scale_log2;
    const int n_tiles = (n_keys + KEYS_PER_TILE - 1) / KEYS_PER_TILE;

    // Row coordinates in the maps. A K tile past the end of the head reads the next head or zeros: masked.
    const int k_row0 = blockIdx.y * n_keys;
    constexpr int V_ROWS_PER_TILE = KEYS_PER_TILE / 2;
    const int v_row0 = blockIdx.y * v8_tokens(n_keys) / (KEYS_PER_TILE / V_ROWS_PER_TILE);
    auto load_k = [&](int buffer, int tile, uint64_t* barrier) {
        tma_load_2d(k_tiles[buffer], k_map, 0, k_row0 + tile * KEYS_PER_TILE, barrier);
    };
    // v_token's tiles lie in one piece per head, padded to whole tiles like V8: a tile is 512 contiguous bytes
    const float4* tok_head = v_token + size_t(blockIdx.y) * v8_tokens(n_keys) / 2;
    auto load_v = [&](int buffer, int tile, uint64_t* barrier) {
        tma_load_2d(v_tiles[buffer], v_map, 0, v_row0 + tile * V_ROWS_PER_TILE, barrier);
        if constexpr (TOKEN_TILES)
            bulk_load(tok_tiles[buffer], tok_head + tile * (KEYS_PER_TILE / 2), TOK_TILE_BYTES, barrier);
    };
    constexpr int V_BYTES = V_TILE_BYTES + (TOKEN_TILES ? TOK_TILE_BYTES : 0);  // what one load_v delivers

    // M4.48: freed[p] completes when all computing warps are done with the K and V that the copies of iterations
    // t ≡ p (mod 2) overwrite: those of iteration t − 1 (for t = 0, K(0) of the prologue's Q·Kᵀ)
    [[maybe_unused]] __shared__ __align__(8) uint64_t freed[2];
    auto release = [&](int p) {
        __syncwarp();
        if (lane == 0) mbarrier_arrive(&freed[p]);
    };
    if (threadIdx.x == 0) {
        for (int i = 0; i < 3; ++i) mbarrier_init(&landed[i], 1);
        if constexpr (UNICA_PRODUCER)
            for (int i = 0; i < 2; ++i) mbarrier_init(&freed[i], WARPS_PER_BLOCK);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    auto prologue_copies = [&] {
        mbarrier_expect_bytes(&landed[2], (n_tiles > 1 ? 2 : 1) * K_TILE_BYTES + V_BYTES);
        load_k(0, 0, &landed[2]);
        load_v(0, 0, &landed[2]);
        if (n_tiles > 1) load_k(1, 1, &landed[2]);
    };
    if constexpr (UNICA_PRODUCER) {
        if (warp >= WARPS_PER_BLOCK) {  // the producer warpgroup: one thread issues, the others leave
            asm volatile("setmaxnreg.dec.sync.aligned.u32 24;" ::: "memory");
            if (threadIdx.x == THREADS_PER_BLOCK) {
                prologue_copies();
                // the copies of iteration t: K(t + 2) and V(t + 1), into the buffers iteration t − 1 read
                for (int t = 0; t + 1 < n_tiles; ++t) {
                    mbarrier_wait(&freed[t % 2], t / 2 % 2);
                    // ldmatrix read these buffers through the generic proxy; TMA writes them through the async one
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_expect_bytes(&landed[t % 2], (t + 2 < n_tiles ? K_TILE_BYTES : 0) + V_BYTES);
                    if (t + 2 < n_tiles) load_k(t % 2, t + 2, &landed[t % 2]);
                    load_v(1 - t % 2, t + 1, &landed[t % 2]);
                }
            }
            return;
        }
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(COMPUTING_REGISTERS) : "memory");
    } else if (threadIdx.x == 0) {
        prologue_copies();
    }
    load_q8_async(q_tiles[warp], Q8 + q_offset, Q8, n_rows, lane);
    cp_async_commit();

    float row_max[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float row_sum[2] = {0.0f, 0.0f};
    float out[D_BLOCKS][4] = {};

    cp_async_wait();
    mbarrier_wait(&landed[2], 0);
    if constexpr (UNICA_PRODUCER)
        __syncwarp();  // the Q tiles of the other lanes (each warp has its own)
    else
        __syncthreads();  // the Q tiles of the other lanes
    float scores[2][KEY_BLOCKS][4];  // tile t in scores[t % 2]; Q·Kᵀ(t + 1) writes the other
    qk_scores_int8_q_smem(scores[0], q_tiles[warp], k_tiles[0], lane);
    if (n_keys < KEYS_PER_TILE) mask_scores(scores[0], col, n_keys);
    if constexpr (UNICA_PRODUCER) release(0);  // done with K(0)

    // As in legacy.cu's attention_int8pv_pipe. Iteration t reads K(t + 1) and V(t), issued in iteration t − 1 (or the
    // prologue), and issues K(t + 2) into the K buffer of t and V(t + 1) into the other V buffer.
    auto iteration = [&](int t, auto parity) {
        constexpr int now = decltype(parity)::value, later = 1 - now;
        const int key0 = t * KEYS_PER_TILE;
        if (t > 0) mbarrier_wait(&landed[later], (t - 1) / 2 % 2);  // its ((t − 1) / 2)-th phase
        if constexpr (!UNICA_PRODUCER) {
            __syncthreads();  // every warp is done with K(t) and V(t − 1)
            if (threadIdx.x == 0 && t + 1 < n_tiles) {
                // ldmatrix read these buffers through the generic proxy; TMA writes them through the async one
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_expect_bytes(&landed[now], (t + 2 < n_tiles ? K_TILE_BYTES : 0) + V_BYTES);
                if (t + 2 < n_tiles) load_k(now, t + 2, &landed[now]);
                load_v(later, t + 1, &landed[now]);
            }
        }

        float alpha[2], beta[2];
        qk_scores_int8_q_smem(scores[later], q_tiles[warp], k_tiles[later], lane);  // stale in the last iteration
        if constexpr (TOKEN_TILES)  // M4.41: landed with V(t), in this tile's buffer
            online_softmax_u8_token<MMA_N / 2>(scores[now], row_max, row_sum, alpha, beta,
                                               q_scale_log2 * k_scale[key0 / K_SCALE_BLOCK], &tok_tiles[now][lane % 4]);
        else
            online_softmax_u8(scores[now], row_max, alpha, beta, q_scale_log2 * k_scale[key0 / K_SCALE_BLOCK]);
        uint32_t p[PV8_STEPS][4];
#pragma unroll
        for (int step = 0; step < PV8_STEPS; ++step) p_fragment_u8(p[step], scores[now], step);
        if constexpr (!UNICA_V_TOKEN) add_row_sum_u8(row_sum, p, alpha, beta);
        pv_int8(out, p, v_tiles[now], alpha, beta, lane);

        // M4.49: the column from a fresh read of the thread index, so that it does not live across the loop
        const int next_valid = n_keys - key0 - KEYS_PER_TILE;  // ≤ 0 in the last iteration: all masked
        if (next_valid < KEYS_PER_TILE) mask_scores(scores[later], 2 * (fresh_thread_index() % 4), next_valid);
        if constexpr (UNICA_PRODUCER) release(later);  // done with K(t + 1) and V(t): the copies of iteration t + 1 may come
    };

    int t = 0;
    for (; t + 1 < n_tiles; t += 2) {
        iteration(t, Parity<0>{});
        iteration(t + 1, Parity<1>{});
    }
    if (t < n_tiles) iteration(t, Parity<0>{});

    if constexpr (UNICA_V_OUTLIERS > 0) add_v_outliers(out, row_max, q_tiles[warp], K8, out_idx, out_v, k_scale, q_scale_log2,
                                                       n_keys, lane);
    // M4.49: lane, row0 and n_rows from a fresh read of the thread index, so that they do not live across the loop
    const int index = fresh_thread_index(), at = index % WARP_SIZE;
    const int first = blockIdx.x * ROWS_PER_BLOCK + index / WARP_SIZE * ROWS_PER_WARP;  // row0
    store_out(O + o_layout.offset(blockIdx.y, first), out, row_sum, at / 4, 2 * (at % 4),
              min(n_queries - first, ROWS_PER_WARP), v_scale, o_layout.token);
}

// ---------------------------------------------------------------------------------------------------
// M4.8: P·V in NVFP4 (e2m1 values, an e4m3 scale per 16 keys), mma.sync m16n8k64 kind::mxf4nvf4, which runs
// at twice the 8-bit rate and accumulates in f32 itself (LOG.md M4.1).
// ---------------------------------------------------------------------------------------------------

constexpr int MMA_K4 = 64;          // keys in one m16n8k64: a whole tile
constexpr int FP4_SCALE_BLOCK = 16;  // keys per e4m3 scale
// A tile of 64 keys is one mma's k (m16n8k64)
// Measured on the card (LOG.md M4.8): lane l holds k = 8 · (l % 4) + 32 · (r / 2) + j of row l / 4 + 8 · (r % 2)
// in nibble j of A register r, and k = 8 · (l % 4) + 32 · r + j of column l / 4 in B register r. Q·Kᵀ leaves
// lane l the scores of keys 8 · b + 2 · (l % 4) + {0, 1}, b = 0..7. Packed in order, blocks 4h .. 4h + 3 into
// A register 2h (+1), they put key 32h + 8 · (j / 2) + 2 · (l % 4) + j % 2 at slot k = 32h + 8 · (l % 4) + j:
__host__ __device__ constexpr int fp4_slot_key(int slot) {
    return 32 * (slot / 32) + 8 * (slot % 8 / 2) + 2 * (slot % 32 / 8) + slot % 2;
}

// V4: per head and tile, HEAD_DIM rows of V4_ROW_BYTES, one per channel. Bytes 0..31 hold the channel's 64
// keys as e2m1 in slot order, slot 2b in the low nibble of byte b: an 8 × 16-byte matrix of ldmatrix (non-
// transposed) gives lane l word l % 4 of row l / 4, which is B register 0 of m16n8k64 (bytes 0..15) or 1
// (bytes 16..31). Bytes 32..47 of rows 0..31 hold the e4m3 scales, 4 per channel (byte i = slots 16i ..
// 16i + 15), placed so that lane l reads the 16 channels l / 4 + 8n it needs with 4 LDS.128: channel d at
// row d % 8 + 8 · (d / 32), word d / 8 % 4. 48-byte rows: the 8 rows of an ldmatrix start in 8 different
// groups of 4 banks.
constexpr int V4_ROW_BYTES = 48;
constexpr int V4_TILE_BYTES = HEAD_DIM * V4_ROW_BYTES;
constexpr int V4_SCALE_OFFSET = 32;

__device__ __forceinline__ int v4_scale_row(int dim) { return dim % 8 + 8 * (dim / 32); }

// Eight floats as e2m1, nearest, ties to even, saturating at ±6: x[j] in nibble j.
__device__ __forceinline__ uint32_t pack_e2m1x8(const float x[8]) {
    uint32_t word;
    asm("{ .reg .b8 b0, b1, b2, b3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
        "mov.b32 %0, {b0, b1, b2, b3}; }"
        : "=r"(word)
        : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
    return word;
}

// x ≥ 0 as an e4m3 byte, nearest, saturating at 448; and the byte back as a float.
__device__ __forceinline__ uint32_t to_e4m3(float x) {
    uint16_t pair;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(pair) : "f"(x));
    return pair & 0xff;
}
__device__ __forceinline__ float from_e4m3(uint32_t byte) {
    uint32_t halves;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(halves) : "h"(uint16_t(byte)));
    return __half2float(__ushort_as_half(uint16_t(halves & 0xffff)));
}

// V for attention_fp4pv: per channel, v ≈ e2m1 · (e4m3 scale per 16 keys) · v_outer[channel], v_outer =
// max |v| of the channel / (6 · 448), so the largest scale is 448. One thread per (head, tile, block of 16
// slots, channel), channels fastest: a warp reads 32 neighbouring channels of the same keys. Keys past n_tokens
// are zeros. Bit for bit bench/emulate.py's block_scaled(..., "nvfp4", outer="channel") on V in slot order
// (tests/test_fp4.py). Grid: ceil(heads · tiles · 4 · HEAD_DIM / QUANTIZE_THREADS).
extern "C" __global__ void __launch_bounds__(QUANTIZE_THREADS)
    quantize_v4(const __nv_bfloat16* V, const float* v_outer, uint8_t* V4, int n_tokens, int n_threads, TokenLayout in) {
    const int index = blockIdx.x * QUANTIZE_THREADS + threadIdx.x;
    if (index >= n_threads) return;
    const int dim = index % HEAD_DIM;
    const int block = index / HEAD_DIM % (MMA_K4 / FP4_SCALE_BLOCK);
    const int tile_of_head = index / (HEAD_DIM * MMA_K4 / FP4_SCALE_BLOCK);  // head · tiles + tile
    const int tiles = (n_tokens + MMA_K4 - 1) / MMA_K4;
    const int head = tile_of_head / tiles, tile = tile_of_head % tiles;
    const float outer = v_outer[head * HEAD_DIM + dim];

    float values[FP4_SCALE_BLOCK];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < FP4_SCALE_BLOCK; ++i) {
        const int key = tile * MMA_K4 + fp4_slot_key(block * FP4_SCALE_BLOCK + i);
        values[i] = key < n_tokens ? __bfloat162float(V[in.offset(head, key) + dim]) : 0.0f;
        amax = fmaxf(amax, fabsf(values[i]));
    }
    // · (1/6), not / 6, as in quantize_int8: torch divides by a scalar through its reciprocal, and the two differ in
    // the last bit, which moves a scale that lies on a tie of e4m3 (152 between 144 and 160) to the other side
    const uint32_t scale_byte = to_e4m3(amax * (1.0f / 6.0f) / outer);
    const float scale = from_e4m3(scale_byte) * outer;
#pragma unroll
    for (int i = 0; i < FP4_SCALE_BLOCK; ++i) values[i] = scale > 0.0f ? values[i] / scale : 0.0f;

    uint8_t* row = V4 + size_t(tile_of_head) * V4_TILE_BYTES;
    *reinterpret_cast<uint2*>(row + dim * V4_ROW_BYTES + block * 8) = make_uint2(pack_e2m1x8(values), pack_e2m1x8(values + 8));
    row[v4_scale_row(dim) * V4_ROW_BYTES + V4_SCALE_OFFSET + 4 * (dim / 8 % 4) + block] = uint8_t(scale_byte);
}

constexpr float P_LOG2_GAIN = 10.0f;  // P is taken times 2^10: block scales of P up to 1024 / 6 stay below 448
constexpr uint32_t E2M1_ONES = 0x22222222, UE4M3_ONES = 0x38383838;  // 1.0 in every nibble, in every byte

// out += A · Bᵀ, A e2m1 16 × 64 and B e2m1 8 × 64, each block of 16 k times its e4m3 scale, f32 accumulate.
// scale_a: in lane 4g the 4 scales of row g, in lane 4g + 1 those of row g + 8; scale_b: in lane 4g those of
// column g; byte i for k 16i .. 16i + 15 (LOG.md M4.8). The other lanes' scale registers are not read.
__device__ __forceinline__ void mma_nvf4_16864(float acc[4], const uint32_t a[4], const uint32_t b[2], uint32_t scale_a,
                                               uint32_t scale_b) {
    asm("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(scale_a), "r"(scale_b));
}

__device__ __forceinline__ float lg2_approx(float x) {
    float y;
    asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// Online softmax for an NVFP4 P. P is taken against the running max (no beta: the block scales keep the range,
// and the mma adds straight into out), times 2^P_LOG2_GAIN. Each block of 16 keys gets the scale
// e4m3(max P / 6), and P̂ = P / scale comes out of the exponent itself: 2^(s · scale_log2 + offset) with
// offset = gain − max − log2(scale). The block maxes are the first two steps of the row-max tree: a lane has 8
// scores of a row in each half h of the tile, its neighbour (lane ^ 1) the other 8 of the same block, and lane ^ 2
// the other block of that half. p gets the A fragment, scale_p the A scale register of lanes 4g and 4g + 1.
__device__ __forceinline__ void online_softmax_fp4(const float scores[KEY_BLOCKS][4], float row_max[2], float alpha[2],
                                                   uint32_t p[4], uint32_t& scale_p, float scale_log2, int lane) {
    float block_max[2][2];  // [half][h]: this lane's block of row g + 8 · half in half h of the tile
    float tile_max[2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float m = -CUDART_INF_F;
#pragma unroll
            for (int block = 4 * h; block < 4 * h + 4; ++block)
                m = fmaxf(m, fmaxf(scores[block][2 * half], scores[block][2 * half + 1]));
            block_max[half][h] = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
        }
        const float mine = fmaxf(block_max[half][0], block_max[half][1]);
        tile_max[half] = fmaxf(mine, __shfl_xor_sync(0xffffffff, mine, 2)) * scale_log2;
    }
    uint32_t own_scales = 0;  // byte 2 · half + h
    float offset[2][2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const float new_max = fmaxf(row_max[half], tile_max[half]);
        alpha[half] = exp2_ftz(row_max[half] - new_max);
        row_max[half] = new_max;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float block_p = exp2_ftz(fmaf(block_max[half][h], scale_log2, P_LOG2_GAIN - new_max));
            const uint32_t byte = to_e4m3(block_p * (1.0f / 6.0f));
            // a scale of 0 (the block's P below 2^-10 · 6 / 1024) zeroes the block in the mma; FLT_MIN keeps the
            // offset finite, so a masked score (−inf) still gives 0, not NaN
            offset[half][h] = P_LOG2_GAIN - new_max - lg2_approx(fmaxf(from_e4m3(byte), 1.17549435e-38f));
            own_scales |= byte << (8 * (2 * half + h));
        }
    }
    // lane 4g needs row g's scales of blocks 0..3 = (h0, own pair), (h0, other), (h1, own), (h1, other); lane
    // 4g + 1 the same of row g + 8. The other pair of keys is in lane ^ 2.
    const uint32_t other_scales = __shfl_xor_sync(0xffffffff, own_scales, 2);
    scale_p = __byte_perm(own_scales, other_scales, lane & 1 ? 0x7362 : 0x5140);
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int half = r % 2, h = r / 2;
        float x[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) x[j] = exp2_ftz(fmaf(scores[4 * h + j / 2][2 * half + j % 2], scale_log2, offset[half][h]));
        p[r] = pack_e2m1x8(x);
    }
}

// out = out · alpha + P̂ · V̂ for one tile, V̂ in V4 rows in shared memory. One ldmatrix.x4 gives B for two blocks of 8
// channels; lane l reads the scales of channels l / 4 + 8n with 4 LDS.128. A 17th mma against a column of ones
// gathers Σ P̂ · scale into row_sum_acc (row g in lane 4g's [0], row g + 8 in its [2], 0 elsewhere): the
// denominator over exactly the P the numerator used, rescaled with out.
__device__ __forceinline__ void pv_fp4(float out[D_BLOCKS][4], float row_sum_acc[4], const uint32_t p[4], uint32_t scale_p,
                                       const uint8_t* v_tile, const float alpha[2], int lane) {
    if (!__all_sync(0xffffffff, alpha[0] == 1.0f && alpha[1] == 1.0f)) {
        rescale_out(out, alpha);
        row_sum_acc[0] *= alpha[0];
        row_sum_acc[1] *= alpha[0];
        row_sum_acc[2] *= alpha[1];
        row_sum_acc[3] *= alpha[1];
    }
    uint32_t scale_v[D_BLOCKS];
#pragma unroll
    for (int n = 0; n < 4; ++n) {
        const uint4 words = *reinterpret_cast<const uint4*>(v_tile + (lane / 4 + 8 * n) * V4_ROW_BYTES + V4_SCALE_OFFSET);
        scale_v[4 * n] = words.x;
        scale_v[4 * n + 1] = words.y;
        scale_v[4 * n + 2] = words.z;
        scale_v[4 * n + 3] = words.w;
    }
    // matrices: rows 16i .. 16i + 7 bytes 0..15 and 16..31, then rows 16i + 8 .. 16i + 15 likewise
    const uint8_t* lane_row = v_tile + (lane % 8 + 8 * (lane / 16)) * V4_ROW_BYTES + 16 * (lane / 8 % 2);
#pragma unroll
    for (int pair = 0; pair < D_BLOCKS / 2; ++pair) {
        uint32_t b[4];
        ldmatrix_x4(b, lane_row + 16 * pair * V4_ROW_BYTES);
        const uint32_t left[2] = {b[0], b[1]}, right[2] = {b[2], b[3]};
        mma_nvf4_16864(out[2 * pair], p, left, scale_p, scale_v[2 * pair]);
        mma_nvf4_16864(out[2 * pair + 1], p, right, scale_p, scale_v[2 * pair + 1]);
    }
    const uint32_t ones[2] = {lane < 4 ? E2M1_ONES : 0u, lane < 4 ? E2M1_ONES : 0u};  // column 0 only
    mma_nvf4_16864(row_sum_acc, p, ones, scale_p, UE4M3_ONES);
}

// ---------------------------------------------------------------------------------------------------
// M4.12 (H9): Q·Kᵀ in NVFP4 too. Q and K as e2m1 with an e4m3 scale per 16 channels of d and one float scale per
// head; m16n8k64 sums over d, so a key tile takes 2 k-steps × 8 blocks of keys = 16 OMMA instead of 32 IMMA.
// ---------------------------------------------------------------------------------------------------

// K4 / Q4: one row per token, d in order as e2m1 (d 2b in the low nibble of byte b), then the 8 e4m3 scales of the
// 16-channel blocks (byte 64 + i for d 16i .. 16i + 15). A 16-byte matrix of ldmatrix gives lane l word l % 4 of row
// l / 4: bytes 0..15 of 8 keys are B register 0 of the k-step over d 0..63, bytes 16..31 register 1, bytes 32..63 the
// same for d 64..127. Word s of the scales (bytes 64 + 4s ..) is the scale register of k-step s. 80-byte rows: the 8
// rows of an ldmatrix start in 8 different groups of 4 banks, and a TMA box is a multiple of 16 bytes.
constexpr int FP4_ROW_BYTES = 80;
constexpr int FP4_SCALES_AT = 64;
constexpr int D_BLOCKS_FP4 = HEAD_DIM / FP4_SCALE_BLOCK;  // 8 scales per token
constexpr int K4_TILE_BYTES = KEYS_PER_TILE * FP4_ROW_BYTES;

// Q or K for attention_fp4: x − mean (K: softmax cancels the mean) ≈ e2m1 · (e4m3 scale per 16 channels) · outer[head],
// outer = max |x − mean| over the head / (6 · 448). One thread per (token, block of 16 channels). Bit for bit
// bench/emulate.py's block_scaled(x − mean, "nvfp4", -1) (tests/test_fp4.py). mean: float [heads][mean rows][HEAD_DIM]
// or null, one row per mean_tokens tokens (M4.21, Q smoothing; mean_tokens = n_tokens: one per head).
// X4 has padded_tokens ≥ n_tokens rows per head; the rows past n_tokens are not written.
// Grid: ceil(heads · n_tokens · 8 / QUANTIZE_THREADS).
extern "C" __global__ void __launch_bounds__(QUANTIZE_THREADS)
    quantize_fp4_rows(const __nv_bfloat16* X, const float* mean, const float* outer, uint8_t* X4, int n_tokens,
                      int padded_tokens, int n_threads, TokenLayout in, const __half* freqs, long long freqs_batch,
                      int mean_tokens) {
    const int index = blockIdx.x * QUANTIZE_THREADS + threadIdx.x;
    if (index >= n_threads) return;
    const int block = index % D_BLOCKS_FP4;
    const size_t row = index / D_BLOCKS_FP4;  // head · n_tokens + token
    const int head = int(row / n_tokens);
    const int token = int(row % n_tokens);
    if (mean) mean += (size_t(head) * ((n_tokens + mean_tokens - 1) / mean_tokens) + token / mean_tokens) * HEAD_DIM;
    float values[FP4_SCALE_BLOCK];
    float amax = 0.0f;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        float chunk[CHUNK];
        const int dim = block * FP4_SCALE_BLOCK + half * CHUNK;
        load_chunk_rope(chunk, X + in.offset(head, token) + dim, mean ? mean + dim : nullptr,
                        freqs ? freqs + size_t(head / in.heads) * freqs_batch + size_t(token) * ROPE_HALVES_PER_TOKEN : nullptr,
                        dim);
#pragma unroll
        for (int i = 0; i < CHUNK; ++i) {
            values[half * CHUNK + i] = chunk[i];
            amax = fmaxf(amax, fabsf(chunk[i]));
        }
    }
    const uint32_t scale_byte = to_e4m3(amax * (1.0f / 6.0f) / outer[head]);  // · (1/6): as torch, M4.8
    const float scale = from_e4m3(scale_byte) * outer[head];
#pragma unroll
    for (int i = 0; i < FP4_SCALE_BLOCK; ++i) values[i] = scale > 0.0f ? values[i] / scale : 0.0f;
    uint8_t* out = X4 + (size_t(head) * padded_tokens + token) * FP4_ROW_BYTES;  // rows of each head padded
    *reinterpret_cast<uint2*>(out + block * 8) = make_uint2(pack_e2m1x8(values), pack_e2m1x8(values + 8));
    out[FP4_SCALES_AT + block] = uint8_t(scale_byte);
}

// M4.21: Q smoothing in attention_fp4. 0: off. n: Q4 holds q − q̄_B, q̄_B the mean of Q over its block B of n queries,
// and since q·k = (q − q̄_B)·k + q̄_B·k, the scores get q̄_B·k back, exactly: smooth_bias writes it per (B, key), and a
// tile's Q·Kᵀ accumulators start from it instead of 0. The FP4 values then only carry what varies within B, which is
// smaller than q itself. A block of the kernel's queries lies within one B.
#ifndef UNICA_SMOOTH_Q
#define UNICA_SMOOTH_Q 0
#endif
constexpr int SMOOTH_Q = UNICA_SMOOTH_Q;
static_assert(SMOOTH_Q % ROWS_PER_BLOCK == 0, "a block of the kernel must lie within one smoothing block");
constexpr int BIAS_TILE_BYTES = SMOOTH_Q ? KEYS_PER_TILE * int(sizeof(float)) : 0;

// The scores' share of Q's block means, q̄_B·(k − k̄) / qk_outer, in the units of attention_fp4's Q·Kᵀ accumulators
// (k − k̄, not k: q̄_B·k̄ is one number per row, which softmax cancels, and the bias stays small). From the bf16 K
// rotated on the way in like the quantizers (freqs, or null), not from the quantized K. k_mean: float
// [heads][HEAD_DIM]; q_hi, q_lo: the means as two bf16, q̄ ≈ hi + lo, mean_rows ≥ n_blocks of them per head (a multiple
// of 16, zero rows past n_blocks), already in the mma's A fragments: [heads][mean_rows / 16][8 k-steps][32 lanes] of
// 4 registers, one coalesced LDG.128 per lane and k-step (unica.smooth_bias); bias: float [heads][n_blocks][padded_keys],
// 0 past n_keys.
//
// A GEMM [n_blocks × 128] · [128 × keys] per head on the tensor cores, near float: (k − k̄) is split the same way,
// and q̄·x ≈ hi·hi + hi·lo + lo·hi, three bf16 mma with an f32 accumulator (the lo·lo term left out is ~2^-18
// relative). On the FP32 pipe it was 2.2 ms at best, bound by shared memory: every FMA needs an operand from there.
// A block of threads splits BIAS_KEYS keys into shared memory (rows padded by 16 bytes, so ldmatrix's 8 rows hit
// 8 different banks); each warp holds the B fragments of its 16 keys in registers and goes through the means 16 at a
// time, their A fragments read from global memory (all blocks of a head read the same means: L1 and L2).
// Grid: (ceil(padded_keys / BIAS_KEYS), heads). Block: BIAS_THREADS.
constexpr int BIAS_THREADS = 128;
constexpr int BIAS_KEYS = BIAS_THREADS / WARP_SIZE * 16;
constexpr int BIAS_ROW = HEAD_DIM + 8;  // bf16
__device__ __forceinline__ uint32_t split_bf16x2(float x0, float x1, uint32_t& low) {
    const __nv_bfloat162 high = __floats2bfloat162_rn(x0, x1);
    const float2 back = __bfloat1622float2(high);
    const __nv_bfloat162 rest = __floats2bfloat162_rn(x0 - back.x, x1 - back.y);
    low = *reinterpret_cast<const uint32_t*>(&rest);
    return *reinterpret_cast<const uint32_t*>(&high);
}
extern "C" __global__ void __launch_bounds__(BIAS_THREADS)
    smooth_bias(const __nv_bfloat16* K, const float* k_mean, const uint4* q_hi, const uint4* q_lo,
                const float* qk_outer, float* bias, int n_keys, int padded_keys, int n_blocks, int mean_rows,
                TokenLayout in, const __half* freqs, long long freqs_batch) {
    __shared__ __align__(16) __nv_bfloat16 k_hi[BIAS_KEYS][BIAS_ROW], k_lo[BIAS_KEYS][BIAS_ROW];
    const int head = blockIdx.y, key0 = blockIdx.x * BIAS_KEYS;
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    const __half* table = freqs ? freqs + size_t(head / in.heads) * freqs_batch : nullptr;
    for (int c = threadIdx.x; c < BIAS_KEYS * CHUNKS_PER_ROW; c += BIAS_THREADS) {
        const int key = c / CHUNKS_PER_ROW, dim = c % CHUNKS_PER_ROW * CHUNK;
        float values[CHUNK] = {};
        if (key0 + key < n_keys)
            load_chunk_rope(values, K + in.offset(head, key0 + key) + dim, k_mean + head * HEAD_DIM + dim,
                            table ? table + size_t(key0 + key) * ROPE_HALVES_PER_TOKEN : nullptr, dim);
        uint4 high, low;
        high.x = split_bf16x2(values[0], values[1], low.x);
        high.y = split_bf16x2(values[2], values[3], low.y);
        high.z = split_bf16x2(values[4], values[5], low.z);
        high.w = split_bf16x2(values[6], values[7], low.w);
        *reinterpret_cast<uint4*>(&k_hi[key][dim]) = high;
        *reinterpret_cast<uint4*>(&k_lo[key][dim]) = low;
    }
    __syncthreads();

    // B fragments of the warp's 2 blocks of 8 keys, 8 k-steps of 16 channels: one ldmatrix.x4 gives 2 k-steps
    uint32_t b_hi[2][8][2], b_lo[2][8][2];
#pragma unroll
    for (int n = 0; n < 2; ++n)
#pragma unroll
        for (int s = 0; s < 8; s += 2) {
            const int row = warp * 16 + n * 8 + lane % 8, col = 16 * s + 8 * (lane / 8);
            uint32_t r[4];
            ldmatrix_x4(r, &k_hi[row][col]);
            b_hi[n][s][0] = r[0], b_hi[n][s][1] = r[1], b_hi[n][s + 1][0] = r[2], b_hi[n][s + 1][1] = r[3];
            ldmatrix_x4(r, &k_lo[row][col]);
            b_lo[n][s][0] = r[0], b_lo[n][s][1] = r[1], b_lo[n][s + 1][0] = r[2], b_lo[n][s + 1][1] = r[3];
        }

    const int g = lane / 4, t = 2 * (lane % 4);
    const float inverse_outer = 1.0f / qk_outer[head];
    for (int m0 = 0; m0 < n_blocks; m0 += 16) {
        float acc[2][4] = {};
        const size_t fragments = (size_t(head) * mean_rows + m0) / 16 * 8 * WARP_SIZE;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            uint32_t a_hi[4], a_lo[4];
            *reinterpret_cast<uint4*>(a_hi) = __ldg(q_hi + fragments + s * WARP_SIZE + lane);
            *reinterpret_cast<uint4*>(a_lo) = __ldg(q_lo + fragments + s * WARP_SIZE + lane);
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                mma_bf16_16816(acc[n], a_hi, b_hi[n][s]);
                mma_bf16_16816(acc[n], a_hi, b_lo[n][s]);
                mma_bf16_16816(acc[n], a_lo, b_hi[n][s]);
            }
        }
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            const int key = key0 + warp * 16 + n * 8 + t;
#pragma unroll
            for (int h = 0; h < 2; ++h) {  // rows g and g + 8
                const int block = m0 + g + 8 * h;
                if (block >= n_blocks || key >= padded_keys) continue;
                const float2 out = make_float2(key < n_keys ? acc[n][2 * h] * inverse_outer : 0.0f,
                                               key + 1 < n_keys ? acc[n][2 * h + 1] * inverse_outer : 0.0f);
                *reinterpret_cast<float2*>(&bias[(size_t(head) * n_blocks + block) * padded_keys + key]) = out;
            }
        }
    }
}

// Scores of one tile: 8 blocks of 8 keys, each 2 k-steps of 64 channels. q: Q's A fragments of both steps, held in
// registers for the whole kernel; q_scale: their scale registers. One ldmatrix.x4 gives a block's B registers of
// both steps; one LDS.64 its two scale registers (lane 4g: key 8b + g). bias_tile: the tile's smooth_bias values, which
// the accumulators start from (SMOOTH_Q; lane l holds keys 8b + 2 · (l % 4) and + 1 in both of its rows).
__device__ __forceinline__ void qk_scores_fp4(float scores[KEY_BLOCKS][4], const uint32_t q[2][4], const uint32_t q_scale[2],
                                              const uint8_t* k_tile, const float* bias_tile, int lane) {
    const uint8_t* lane_row = k_tile + (lane % 8) * FP4_ROW_BYTES + 16 * (lane / 8);
#pragma unroll
    for (int block = 0; block < KEY_BLOCKS; ++block) {
        uint32_t b[4];
        ldmatrix_x4(b, lane_row + block * MMA_N * FP4_ROW_BYTES);
        const uint2 k_scale = *reinterpret_cast<const uint2*>(k_tile + (block * MMA_N + lane / 4) * FP4_ROW_BYTES + FP4_SCALES_AT);
        const uint32_t low[2] = {b[0], b[1]}, high[2] = {b[2], b[3]};
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if constexpr (SMOOTH_Q > 0) {
            // the same pair into both halves of the accumulator quad: two LDS.64 (ld.volatile, which ptxas may not
            // merge into one). One LDS.64 and two MOV into the quad: +0.5 ms (M4.21)
            const uint32_t at = smem_address(bias_tile + block * MMA_N + 2 * (lane % 4));
            asm volatile("ld.volatile.shared.v2.f32 {%0, %1}, [%2];" : "=f"(acc[0]), "=f"(acc[1]) : "r"(at));
            asm volatile("ld.volatile.shared.v2.f32 {%0, %1}, [%2];" : "=f"(acc[2]), "=f"(acc[3]) : "r"(at));
        }
        mma_nvf4_16864(acc, q[0], low, q_scale[0], k_scale.x);
        mma_nvf4_16864(acc, q[1], high, q_scale[1], k_scale.y);
#pragma unroll
        for (int i = 0; i < 4; ++i) scores[block][i] = acc[i];
    }
}

// M4.13 (H9b): how many tiles of K and of V attention_fp4 keeps in shared memory. Copies are issued STAGES − 1
// iterations ahead: with 2 (M4.12) a tile's data had one iteration, ~1 µs, to arrive, which a short FP4 tile outruns.
constexpr int STAGES = 2;
constexpr int AHEAD = STAGES - 1;

// M4.21: tiles of smooth_bias per copy. 1: a tile's 256 bytes travel in K's group, one more bulk copy per tile, and
// that copy, not its bytes, cost ~2 ms of the +4.5 (neither an L2 prefetch nor a deeper pipeline helped, nor issuing it
// from another warp). n: one copy every n tiles into one of two buffers (2n · 256 bytes), issued n tiles ahead;
// 8: +3.2 ms, 32: +2.3 ms, the rest is the 16 LDS a tile (ncu: +24 instructions a tile, no MIO throttle).
constexpr int BIAS_CHUNK_TILES = 32;
constexpr int BIAS_SMEM_FLOATS = !SMOOTH_Q ? 1 : (BIAS_CHUNK_TILES == 1 ? STAGES : 2 * BIAS_CHUNK_TILES) * KEYS_PER_TILE;
constexpr int BIAS_GROUP_BYTES = BIAS_CHUNK_TILES == 1 ? BIAS_TILE_BYTES : 0;  // bias bytes in a K/V group

// M4.12: attention_fp4pv with Q·Kᵀ in NVFP4. k_map: K4 (quantize_fp4_rows on K − mean) as rows of FP4_ROW_BYTES, box
// FP4_ROW_BYTES × KEYS_PER_TILE. Q4 as rows of FP4_ROW_BYTES, read once into registers. qk_outer[head]: the product of
// Q's and K's float scales, folded into scale_log2. v_map and v_outer as attention_fp4pv.
//
// Copies (M4.13): iteration j reads K(j + 1) (the Q·Kᵀ of the next tile) and V(j), group j of the copies, which lands
// on full[j % STAGES]. K(j + 1) goes to slot (j + 1) % STAGES and V(j) to slot j % STAGES. Iteration t issues group
// t + AHEAD: its K slot is t % STAGES, which held K(t), read by iteration t − 1; its V slot is (t − 1) % STAGES, which
// held V(t − 1), read by iteration t − 1; its barrier was last waited on in iteration t − 1. All of that is behind the
// __syncthreads at the top of iteration t. K(0), for the Q·Kᵀ before the loop, lands on first_k.
//
// q_bias (SMOOTH_Q, else null): smooth_bias's [heads][query blocks][n_tiles · KEYS_PER_TILE]. With BIAS_CHUNK_TILES = 1
// a tile's 256 bytes travel with the K tile, in the same group. With n: chunk c (tiles nc .. nc + n − 1) lands in
// buffer c % 2 on bias_full[c % 2]. Iteration t scores tile t + 1; when that starts chunk c, it issues chunk c + 1,
// into the buffer of chunk c − 1, whose last tile iteration t − 1 scored (behind the __syncthreads), and waits for
// chunk c, issued n iterations before.
extern "C" __global__ void __launch_bounds__(THREADS_PER_BLOCK, 1)
    attention_fp4(const __grid_constant__ CUtensorMap k_map, const __grid_constant__ CUtensorMap v_map,
                  const uint8_t* K4, const uint8_t* V4, const uint8_t* Q4, const float* qk_outer, const float* v_outer,
                  const float* q_bias, __nv_bfloat16* O, int n_queries, int n_keys, float scale_log2,
                  TokenLayout o_layout) {
    __shared__ __align__(128) uint8_t k_tiles[STAGES][K4_TILE_BYTES];
    __shared__ __align__(128) uint8_t v_tiles[STAGES][V4_TILE_BYTES];
    __shared__ __align__(16) float bias_smem[BIAS_SMEM_FLOATS];
    __shared__ __align__(8) uint64_t full[STAGES];
    __shared__ __align__(8) uint64_t first_k;
    __shared__ __align__(8) uint64_t bias_full[2];

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int row = lane / 4;
    const int col = 2 * (lane % 4);

    const int row0 = blockIdx.x * ROWS_PER_BLOCK + warp * ROWS_PER_WARP;
    const int n_rows = min(n_queries - row0, ROWS_PER_WARP);
    v_outer += size_t(blockIdx.y) * HEAD_DIM;
    const float score_scale = qk_outer[blockIdx.y] * scale_log2;
    const int n_tiles = (n_keys + KEYS_PER_TILE - 1) / KEYS_PER_TILE;

    // Q's A fragments: register r of step s holds row g + 8 · (r % 2), d = 64s + 8 · (lane % 4) + 32 · (r / 2) + j.
    // The scale registers: lane 4g row g, lane 4g + 1 row g + 8 (the others are not read). Rows past n_queries: 0.
    uint32_t q[2][4], q_scale[2];
    {
        const uint8_t* q_rows = Q4 + (size_t(blockIdx.y) * n_queries + row0) * FP4_ROW_BYTES;
#pragma unroll
        for (int s = 0; s < 2; ++s) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int q_row = row + 8 * (r % 2);
                q[s][r] = q_row < n_rows ? *reinterpret_cast<const uint32_t*>(q_rows + q_row * FP4_ROW_BYTES + 32 * s +
                                                                             4 * (lane % 4) + 16 * (r / 2))
                                         : 0u;
            }
            const int scale_row = row + 8 * (lane % 2);
            q_scale[s] = scale_row < n_rows
                             ? *reinterpret_cast<const uint32_t*>(q_rows + scale_row * FP4_ROW_BYTES + FP4_SCALES_AT + 4 * s)
                             : 0u;
        }
    }

    constexpr int smooth_q = SMOOTH_Q > 0 ? SMOOTH_Q : 1;  // as a divisor, for the build without smoothing
    const float* bias_row = nullptr;                       // this block's row of q_bias
    if constexpr (SMOOTH_Q > 0)
        bias_row = q_bias + (size_t(blockIdx.y) * ((n_queries + smooth_q - 1) / smooth_q) +
                             blockIdx.x * ROWS_PER_BLOCK / smooth_q) * n_tiles * KEYS_PER_TILE;
    // where a tile's bias lies in shared memory
    auto bias_tile = [&](int tile) -> const float* {
        if constexpr (BIAS_CHUNK_TILES == 1) return bias_smem + tile % STAGES * KEYS_PER_TILE;
        return bias_smem + (tile / BIAS_CHUNK_TILES % 2 * BIAS_CHUNK_TILES + tile % BIAS_CHUNK_TILES) * KEYS_PER_TILE;
    };
    // M4.14 (H9c): a K4 or V4 tile copied as one contiguous block (cp.async.bulk, 1D), not as a 2D TMA box of 64 rows
    // of 80 bytes and 128 rows of 48: both tiles lie in one piece in global memory, K4 padded to whole tiles
    auto load_k = [&](int buffer, int tile, uint64_t* barrier) {
        bulk_load(k_tiles[buffer], K4 + (size_t(blockIdx.y) * n_tiles + tile) * K4_TILE_BYTES, K4_TILE_BYTES, barrier);
        if constexpr (SMOOTH_Q > 0 && BIAS_CHUNK_TILES == 1)
            bulk_load(bias_smem + buffer * KEYS_PER_TILE, bias_row + size_t(tile) * KEYS_PER_TILE, BIAS_TILE_BYTES, barrier);
    };
    // chunk c of the bias, if there is one; thread 0 only
    auto issue_bias = [&](int c) {
        const int tile0 = c * BIAS_CHUNK_TILES;
        if (tile0 >= n_tiles) return;
        const int bytes = min(BIAS_CHUNK_TILES, n_tiles - tile0) * BIAS_TILE_BYTES;
        mbarrier_expect_bytes(&bias_full[c % 2], bytes);
        bulk_load(bias_smem + c % 2 * BIAS_CHUNK_TILES * KEYS_PER_TILE, bias_row + size_t(tile0) * KEYS_PER_TILE, bytes,
                  &bias_full[c % 2]);
    };
    constexpr bool CHUNKED = SMOOTH_Q > 0 && BIAS_CHUNK_TILES > 1;
    auto load_v = [&](int buffer, int tile, uint64_t* barrier) {
        bulk_load(v_tiles[buffer], V4 + (size_t(blockIdx.y) * n_tiles + tile) * V4_TILE_BYTES, V4_TILE_BYTES, barrier);
    };

    // group j: K(j + 1), if there is one, and V(j); thread 0 only
    auto issue_group = [&](int j) {
        mbarrier_expect_bytes(&full[j % STAGES], (j + 1 < n_tiles ? K4_TILE_BYTES + BIAS_GROUP_BYTES : 0) + V4_TILE_BYTES);
        if (j + 1 < n_tiles) load_k((j + 1) % STAGES, j + 1, &full[j % STAGES]);
        load_v(j % STAGES, j, &full[j % STAGES]);
    };
    if (threadIdx.x == 0) {
        for (int i = 0; i < STAGES; ++i) mbarrier_init(&full[i], 1);
        mbarrier_init(&first_k, 1);
        if constexpr (CHUNKED) mbarrier_init(&bias_full[0], 1), mbarrier_init(&bias_full[1], 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        mbarrier_expect_bytes(&first_k, K4_TILE_BYTES + BIAS_GROUP_BYTES);
        load_k(0, 0, &first_k);
        if constexpr (CHUNKED) issue_bias(0), issue_bias(1);
        for (int j = 0; j < AHEAD && j < n_tiles; ++j) issue_group(j);
    }

    float row_max[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float row_sum_acc[4] = {};
    float out[D_BLOCKS][4] = {};

    mbarrier_wait(&first_k, 0);
    if constexpr (CHUNKED) mbarrier_wait(&bias_full[0], 0);
    float scores[2][KEY_BLOCKS][4];
    qk_scores_fp4(scores[0], q, q_scale, k_tiles[0], bias_tile(0), lane);
    if (n_keys < KEYS_PER_TILE) mask_scores(scores[0], col, n_keys);

    auto iteration = [&](int t, auto parity) {
        constexpr int now = decltype(parity)::value, later = 1 - now;  // the scores, still in two sets
        const int key0 = t * KEYS_PER_TILE;
        mbarrier_wait(&full[t % STAGES], t / STAGES % 2);  // K(t + 1) and V(t)
        __syncthreads();  // every warp is done with iteration t − 1, whose buffers group t + AHEAD reuses
        if (threadIdx.x == 0 && t + AHEAD < n_tiles) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            issue_group(t + AHEAD);
        }
        if constexpr (CHUNKED)
            if ((t + 1) % BIAS_CHUNK_TILES == 0 && t + 1 < n_tiles) {  // tile t + 1 starts chunk c
                const int c = (t + 1) / BIAS_CHUNK_TILES;
                if (threadIdx.x == 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    issue_bias(c + 1);
                }
                mbarrier_wait(&bias_full[c % 2], c / 2 % 2);
            }

        float alpha[2];
        uint32_t p[4], scale_p;
        qk_scores_fp4(scores[later], q, q_scale, k_tiles[(t + 1) % STAGES], bias_tile(t + 1),
                      lane);  // stale in the last iteration
        online_softmax_fp4(scores[now], row_max, alpha, p, scale_p, score_scale, lane);
        pv_fp4(out, row_sum_acc, p, scale_p, v_tiles[t % STAGES], alpha, lane);

        const int next_valid = n_keys - key0 - KEYS_PER_TILE;
        if (next_valid < KEYS_PER_TILE) mask_scores(scores[later], col, next_valid);
    };

    int t = 0;
    for (; t + 1 < n_tiles; t += 2) {
        iteration(t, Parity<0>{});
        iteration(t + 1, Parity<1>{});
    }
    if (t < n_tiles) iteration(t, Parity<0>{});

    const float row_sum[2] = {row_sum_acc[0], row_sum_acc[2]};
    store_out(O + o_layout.offset(blockIdx.y, row0), out, row_sum, row, col, n_rows, v_outer, o_layout.token);
}

