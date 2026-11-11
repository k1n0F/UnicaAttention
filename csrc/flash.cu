/*
 * M1: plain flash attention on mma.sync. Q·Kᵀ in BF16, P·V in FP16, both accumulate in FP32.
 *
 * Built up one piece at a time: one mma, a Q·Kᵀ tile, online softmax, the P·V loop, the whole
 * attention. Each piece keeps its own test kernel, so a broken layout shows up where it was made.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cstdint>

// Shape of one mma.sync.m16n8k16: D[16×8] += A[16×16] · B[16×8].
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

// Tile shape. unica.attention(..., KEYS_PER_TILE=32) and the like rebuild this file with -D to measure
// other shapes; the step kernels and their tests assume the defaults.
#ifndef UNICA_WARPS_PER_BLOCK
#define UNICA_WARPS_PER_BLOCK 4
#endif
#ifndef UNICA_KEYS_PER_TILE
#define UNICA_KEYS_PER_TILE 64
#endif
#ifndef UNICA_MIN_BLOCKS_PER_SM
#define UNICA_MIN_BLOCKS_PER_SM 1
#endif
// M3.5: 0 brings back exp2f in online_softmax_u8, to measure the difference in one run.
#ifndef UNICA_FAST_EXP2
#define UNICA_FAST_EXP2 1
#endif
// M3.9: 0 brings back the multiply of out by alpha in every tile of the u8 kernels.
#ifndef UNICA_LAZY_RESCALE
#define UNICA_LAZY_RESCALE 1
#endif
// M3.13: 0 keeps Q in registers in attention_int8pv_pipe.
#ifndef UNICA_Q_SMEM
#define UNICA_Q_SMEM 1
#endif
// M3.17: 0 gives attention_int8pv_tma V8 in P's order without the pairs (quantize_v layout 1, not 2).
#ifndef UNICA_V_PAIRS
#define UNICA_V_PAIRS 1
#endif
// M4.23: 1 gives attention_int8pv_tma V with one scale per token, folded into P (online_softmax_u8_token).
#ifndef UNICA_V_TOKEN
#define UNICA_V_TOKEN 0
#endif
// M4.41: with UNICA_V_TOKEN, 1 copies each tile's token scales into shared memory with V (one bulk copy on V's
// mbarrier, a tile ahead); 0 reads them from L2 in the tile that uses them (M4.23).
#ifndef UNICA_TOKEN_COPY
#define UNICA_TOKEN_COPY 1
#endif
// M3.15: 0 sums the float P before rounding for the softmax denominator, in the u8 kernels.
#ifndef UNICA_SUM_ROUNDED
#define UNICA_SUM_ROUNDED 1
#endif
// M4.44: n > 0 lets a warp of attention_int8pv_tma skip P·V (and the packing and row sum before it) of a tile in which
// every one of its 16 rows has its tile max below its running max by more than n (log2): every p < 2^−n of the row's largest.
// 0 skips nothing. V per channel only: with V_TOKEN the denominator Σ P' / s is not bounded by the max of P'.
#ifndef UNICA_SKIP_TILES
#define UNICA_SKIP_TILES 0
#endif
static_assert(!(UNICA_SKIP_TILES && UNICA_V_TOKEN), "SKIP_TILES is measured for V per channel only");
// M4.44 H5: n > 0 lets the whole block of attention_int8pv_tma skip a tile (all but its Q·Kᵀ) when every row of all
// its warps lies below its running max by more than n, decided a tile ahead. 0 skips nothing. V per channel only.
#ifndef UNICA_SKIP_BLOCKS
#define UNICA_SKIP_BLOCKS 0
#endif
static_assert(!(UNICA_SKIP_BLOCKS && (UNICA_V_TOKEN || UNICA_SKIP_TILES)), "SKIP_BLOCKS: V per channel, not with SKIP_TILES");
// M4.48: 1 gives attention_int8pv_tma a producer warpgroup: 4 more warps, which drop to 24 registers (setmaxnreg) while
// the 4 computing warps rise to 232; one producer thread issues every copy, and the computing warps never wait for
// each other (no __syncthreads per tile). The host launches 4 more warps.
#ifndef UNICA_PRODUCER
#define UNICA_PRODUCER 0
#endif
// M4.48: 0 brings back .shared::cluster as the destination of the TMA and bulk copies, to measure the difference in
// one run (with it ptxas adds a driver call to every copy, and ignores setmaxnreg: see tma_load_2d)
#ifndef UNICA_CTA_COPIES
#define UNICA_CTA_COPIES 1
#endif
#if UNICA_CTA_COPIES
#define COPY_DESTINATION "shared::cta"
#else
#define COPY_DESTINATION "shared::cluster"
#endif
static_assert(!UNICA_PRODUCER || UNICA_CTA_COPIES, "with .shared::cluster ptxas ignores setmaxnreg");
// M4.49: 1 (the default) makes attention_int8pv_tma read the thread index again for the tail tile's mask and the
// output's store, so that the lane, row, column, row0 and n_rows derived from it do not live (and spill) across
// the tile loop. 0 keeps them, to measure the difference in one run.
#ifndef UNICA_RECOMPUTE
#define UNICA_RECOMPUTE 1
#endif
// M4.51: a suspend-time hint for every mbarrier wait, in nanoseconds (see mbarrier_wait); 0: none
#ifndef UNICA_WAIT_HINT
#define UNICA_WAIT_HINT 0
#endif
static_assert(!(UNICA_PRODUCER && UNICA_SKIP_BLOCKS), "SKIP_BLOCKS shares votes through the per-tile barrier");

constexpr int WARP_SIZE = 32;
constexpr int HEAD_DIM = 128;                      // WAN 2.2
constexpr int ROWS_PER_WARP = 16;                  // query rows: the M of one mma
constexpr int KEYS_PER_TILE = UNICA_KEYS_PER_TILE;  // keys one warp scores per pass

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

// The same mma with fp16 inputs, same layout and same rate. Used for P·V: P lies in [0, 1] and needs
// precision, not range, and fp16 has 10 mantissa bits to bf16's 7.
__device__ __forceinline__ void mma_f16_16816(float acc[4], const uint32_t a[4], const uint32_t b[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
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

// Two neighbouring bf16 as one 32-bit register: one load instead of two.
__device__ __forceinline__ uint32_t load_pair(const __nv_bfloat16* ptr) {
    return *reinterpret_cast<const uint32_t*>(ptr);
}

// The fragments of the layout above, from a row-major tile whose rows are `stride` elements apart.
__device__ __forceinline__ void load_a(uint32_t a[4], const __nv_bfloat16* tile, int stride, int row, int col) {
    a[0] = load_pair(tile + row * stride + col);
    a[1] = load_pair(tile + (row + 8) * stride + col);
    a[2] = load_pair(tile + row * stride + col + 8);
    a[3] = load_pair(tile + (row + 8) * stride + col + 8);
}

__device__ __forceinline__ void load_b(uint32_t b[2], const __nv_bfloat16* tile, int stride, int row, int col) {
    b[0] = load_pair(tile + row * stride + col);
    b[1] = load_pair(tile + row * stride + col + 8);
}

__device__ __forceinline__ void store_acc(float* tile, int stride, const float acc[4], int row, int col) {
    tile[row * stride + col] = acc[0];
    tile[row * stride + col + 1] = acc[1];
    tile[(row + 8) * stride + col] = acc[2];
    tile[(row + 8) * stride + col + 1] = acc[3];
}

// Step 1: one mma, D = A · Bᵀ. Launch with a single warp.
extern "C" __global__ void mma_16x8x16(const __nv_bfloat16* A, const __nv_bfloat16* B, float* D) {
    const int lane = threadIdx.x;
    const int row = lane / 4;
    const int col = 2 * (lane % 4);

    uint32_t a[4], b[2];
    load_a(a, A, MMA_K, row, col);
    load_b(b, B, MMA_K, row, col);
    float acc[4] = {};
    mma_bf16_16816(acc, a, b);
    store_acc(D, MMA_N, acc, row, col);
}

// M3 step 0: one int8 mma, D = A · Bᵀ with A 16 × 32 and B 8 × 32 int8, D 16 × 8 int32. One warp.
extern "C" __global__ void mma_s8_16x8x32(const int8_t* A, const int8_t* B, int* D) {
    const int lane = threadIdx.x;
    const int row = lane / 4;
    const int col = 2 * (lane % 4);
    const int byte = 2 * col;  // 4 int8 per register

    auto word = [](const int8_t* ptr) { return *reinterpret_cast<const uint32_t*>(ptr); };
    const uint32_t a[4] = {word(A + row * 32 + byte), word(A + (row + 8) * 32 + byte),
                           word(A + row * 32 + byte + 16), word(A + (row + 8) * 32 + byte + 16)};
    const uint32_t b[2] = {word(B + row * 32 + byte), word(B + row * 32 + byte + 16)};
    int acc[4] = {};
    mma_s8_16832(acc, a, b);
    D[row * MMA_N + col] = acc[0];
    D[row * MMA_N + col + 1] = acc[1];
    D[(row + 8) * MMA_N + col] = acc[2];
    D[(row + 8) * MMA_N + col + 1] = acc[3];
}

constexpr int Q_STEPS = HEAD_DIM / MMA_K;          // 8 fragments of Q along d
constexpr int KEY_BLOCKS = KEYS_PER_TILE / MMA_N;  // 8 blocks of 16 × 8 in one score tile

// 16 rows of Q as A fragments, kept in registers for the whole pass over K: 32 registers per lane.
// Rows at or past n_rows (the end of the sequence) read as zero.
__device__ __forceinline__ void load_q(uint32_t q[Q_STEPS][4], const __nv_bfloat16* Q, int row, int col,
                                       int n_rows = ROWS_PER_WARP) {
    const bool top = row < n_rows, bottom = row + 8 < n_rows;
#pragma unroll
    for (int step = 0; step < Q_STEPS; ++step) {
        const __nv_bfloat16* tile = Q + step * MMA_K;
        q[step][0] = top ? load_pair(tile + row * HEAD_DIM + col) : 0;
        q[step][1] = bottom ? load_pair(tile + (row + 8) * HEAD_DIM + col) : 0;
        q[step][2] = top ? load_pair(tile + row * HEAD_DIM + col + 8) : 0;
        q[step][3] = bottom ? load_pair(tile + (row + 8) * HEAD_DIM + col + 8) : 0;
    }
}

// Keys at or past n_keys score -inf, so p = 2^(-inf) = 0. In D, the lane's col picks the key.
// M4.49: threadIdx.x read where it is used. asm volatile: ptxas can neither hoist the read out of its branch nor
// merge it with an earlier one, so values derived from it are computed there instead of living across a loop.
__device__ __forceinline__ int fresh_thread_index() {
    int index;
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(index));
    return index;
}

__device__ __forceinline__ void mask_scores(float scores[KEY_BLOCKS][4], int col, int n_keys) {
#pragma unroll
    for (int block = 0; block < KEY_BLOCKS; ++block)
#pragma unroll
        for (int i = 0; i < 4; ++i)
            if (block * MMA_N + col + i % 2 >= n_keys) scores[block][i] = -CUDART_INF_F;
}

// scores = Q · Kᵀ for one tile of KEYS_PER_TILE keys: 8 blocks of 16 × 8, 32 floats per lane.
// The loops must unroll, otherwise scores[block] with a run-time block cannot live in registers
// and goes to local memory.
// MASKED is the last, partial tile: keys at or past n_keys read as zero and score -inf, so p = 0.
// A separate instantiation keeps these checks out of every full tile.
template <bool MASKED = false>
__device__ __forceinline__ void qk_scores(float scores[KEY_BLOCKS][4], const uint32_t q[Q_STEPS][4],
                                          const __nv_bfloat16* K, int row, int col, int n_keys = KEYS_PER_TILE) {
#pragma unroll
    for (int block = 0; block < KEY_BLOCKS; ++block)
#pragma unroll
        for (int i = 0; i < 4; ++i) scores[block][i] = 0.0f;
#pragma unroll
    for (int step = 0; step < Q_STEPS; ++step)
#pragma unroll
        for (int block = 0; block < KEY_BLOCKS; ++block) {
            uint32_t b[2] = {0, 0};
            if (!MASKED || block * MMA_N + row < n_keys)  // in B, the lane's row picks the key
                load_b(b, K + block * MMA_N * HEAD_DIM + step * MMA_K, HEAD_DIM, row, col);
            mma_bf16_16816(scores[block], q[step], b);
        }
    if (MASKED) mask_scores(scores, col, n_keys);
}

// Step 2: S = Q · Kᵀ for 16 query rows and KEYS_PER_TILE keys, summed over HEAD_DIM.
// Q is 16 × HEAD_DIM, K is KEYS_PER_TILE × HEAD_DIM, S is 16 × KEYS_PER_TILE fp32. Launch with a single warp.
extern "C" __global__ void qk_tile(const __nv_bfloat16* Q, const __nv_bfloat16* K, float* S) {
    const int lane = threadIdx.x;
    const int row = lane / 4;
    const int col = 2 * (lane % 4);

    uint32_t q[Q_STEPS][4];
    float scores[KEY_BLOCKS][4];
    load_q(q, Q, row, col);
    qk_scores(scores, q, K, row, col);
#pragma unroll
    for (int block = 0; block < KEY_BLOCKS; ++block)
        store_acc(S + block * MMA_N, KEYS_PER_TILE, scores[block], row, col);
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

// One tile of the online softmax. Turns scores into p = 2^(s · scale_log2 − max) in place, updates the
// running max and this lane's part of the running sum, and returns in alpha how much everything
// accumulated so far shrinks: 2^(old max − new max). Each lane serves two rows, so everything comes
// in pairs: [0] for row (scores[..][0..1]), [1] for row + 8 (scores[..][2..3]).
// scale_log2 = log2(e) / sqrt(HEAD_DIM): exp(x) = 2^(x · log2 e), and ex2 is the fast exponent on the SFU.
__device__ __forceinline__ void online_softmax(float scores[KEY_BLOCKS][4], float row_max[2], float row_sum[2],
                                               float alpha[2], float scale_log2) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        float tile_max = -CUDART_INF_F;
#pragma unroll
        for (int block = 0; block < KEY_BLOCKS; ++block)
            tile_max = fmaxf(tile_max, fmaxf(scores[block][2 * half], scores[block][2 * half + 1]));
        // scale_log2 > 0, so the max of the scaled scores is the scaled max
        const float new_max = fmaxf(row_max[half], quad_max(tile_max) * scale_log2);
        alpha[half] = exp2f(row_max[half] - new_max);  // 2^(-inf) = 0 on the first tile

        float tile_sum = 0.0f;
#pragma unroll
        for (int block = 0; block < KEY_BLOCKS; ++block)
#pragma unroll
            for (int i = 2 * half; i < 2 * half + 2; ++i) {
                scores[block][i] = exp2f(fmaf(scores[block][i], scale_log2, -new_max));
                tile_sum += scores[block][i];
            }
        // all 4 lanes of a row share new_max and alpha, so the partial sums can wait for the end
        row_sum[half] = row_sum[half] * alpha[half] + tile_sum;
        row_max[half] = new_max;
    }
}

// Step 3: online softmax statistics, no V yet. For each of 16 query rows: max + log2(sum) of
// 2^(s · scale_log2) over all keys, with s = q·k. Keys come KEYS_PER_TILE at a time and no
// whole row of S is ever stored.
// n_keys must be a multiple of KEYS_PER_TILE. Launch with a single warp.
extern "C" __global__ void softmax_stats(const __nv_bfloat16* Q, const __nv_bfloat16* K, float* lse,
                                         int n_keys, float scale_log2) {
    const int lane = threadIdx.x;
    const int row = lane / 4;
    const int col = 2 * (lane % 4);

    uint32_t q[Q_STEPS][4];
    load_q(q, Q, row, col);

    float row_max[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float row_sum[2] = {0.0f, 0.0f};  // this lane's part only, summed over the 4 lanes at the end

    for (int key0 = 0; key0 < n_keys; key0 += KEYS_PER_TILE) {
        float scores[KEY_BLOCKS][4], alpha[2];
        qk_scores(scores, q, K + key0 * HEAD_DIM, row, col);
        online_softmax(scores, row_max, row_sum, alpha, scale_log2);
    }

    const float sum_lo = quad_sum(row_sum[0]), sum_hi = quad_sum(row_sum[1]);
    if (lane % 4 == 0) {
        lse[row] = row_max[0] + log2f(sum_lo);
        lse[row + 8] = row_max[1] + log2f(sum_hi);
    }
}

constexpr int D_BLOCKS = HEAD_DIM / MMA_N;        // 16 blocks of 16 × 8 across one row of O
constexpr int KEY_STEPS = KEYS_PER_TILE / MMA_K;  // 4 steps of 16 keys in P·V

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

// Two neighbouring 16 × 8 blocks of P, as Q·Kᵀ left them, are exactly one 16 × 16 A fragment:
// P goes from output to input without leaving the lane. fp16, not bf16: 0.02 % error instead of 0.16 %.
__device__ __forceinline__ void p_fragment(uint32_t p[4], const float scores[KEY_BLOCKS][4], int step) {
    const float* left = scores[2 * step];
    const float* right = scores[2 * step + 1];
    p[0] = pack_f16(left[0], left[1]);
    p[1] = pack_f16(left[2], left[3]);
    p[2] = pack_f16(right[0], right[1]);
    p[3] = pack_f16(right[2], right[3]);
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

// One element of V as float; keys at or past n_keys read as zero. Zero, not just "p = 0": past the
// end of V may lie anything, and 0 · NaN = NaN.
template <bool MASKED>
__device__ __forceinline__ float v_at(const __nv_bfloat16* V, int key, int dim, int n_keys) {
    return !MASKED || key < n_keys ? __bfloat162float(V[key * HEAD_DIM + dim]) : 0.0f;
}

// B fragment of V for 16 keys × 8 columns of d: B[n][k] = V[k][n], converted bf16 -> fp16 for the
// fp16 P·V. The conversion keeps every bf16 mantissa bit; only |v| > 65504 would overflow.
// The two keys of one register sit HEAD_DIM apart in V, so it takes two 16-bit loads instead of one
// 32-bit load. ldmatrix.trans transposes on the way in and replaces this in M2.
template <bool MASKED>
__device__ __forceinline__ void load_v(uint32_t b[2], const __nv_bfloat16* V, int row, int col, int n_keys) {
    b[0] = pack_f16(v_at<MASKED>(V, col, row, n_keys), v_at<MASKED>(V, col + 1, row, n_keys));
    b[1] = pack_f16(v_at<MASKED>(V, col + 8, row, n_keys), v_at<MASKED>(V, col + 9, row, n_keys));
}

// One tile of KEYS_PER_TILE keys for one warp: scores, online softmax, then P·V added into out.
// MASKED: the last tile holds only n_keys keys.
template <bool MASKED>
__device__ __forceinline__ void attention_tile(const uint32_t q[Q_STEPS][4], const __nv_bfloat16* K,
                                               const __nv_bfloat16* V, int n_keys, float row_max[2],
                                               float row_sum[2], float out[D_BLOCKS][4], int row, int col,
                                               float scale_log2) {
    float scores[KEY_BLOCKS][4], alpha[2];
    qk_scores<MASKED>(scores, q, K, row, col, n_keys);
    online_softmax(scores, row_max, row_sum, alpha, scale_log2);

    rescale_out(out, alpha);

#pragma unroll
    for (int step = 0; step < KEY_STEPS; ++step) {
        uint32_t p[4];
        p_fragment(p, scores, step);
        const int key0 = step * MMA_K;
#pragma unroll
        for (int block = 0; block < D_BLOCKS; ++block) {
            uint32_t b[2];
            load_v<MASKED>(b, V + key0 * HEAD_DIM + block * MMA_N, row, col, n_keys - key0);
            mma_f16_16816(out[block], p, b);
        }
    }
}

// Step 4: attention for 16 query rows, O = softmax(Q · Kᵀ / sqrt(HEAD_DIM)) · V, all keys.
// Q is 16 × HEAD_DIM; K and V are n_keys × HEAD_DIM; O is 16 × HEAD_DIM fp32.
// n_keys must be a multiple of KEYS_PER_TILE. Launch with a single warp.
extern "C" __global__ void attention_warp(const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
                                          float* O, int n_keys, float scale_log2) {
    const int lane = threadIdx.x;
    const int row = lane / 4;
    const int col = 2 * (lane % 4);

    uint32_t q[Q_STEPS][4];
    load_q(q, Q, row, col);

    float row_max[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float row_sum[2] = {0.0f, 0.0f};
    float out[D_BLOCKS][4] = {};  // 16 rows × 128 of O: 64 floats per lane

    for (int key0 = 0; key0 < n_keys; key0 += KEYS_PER_TILE)
        attention_tile<false>(q, K + key0 * HEAD_DIM, V + key0 * HEAD_DIM, KEYS_PER_TILE, row_max, row_sum, out,
                              row, col, scale_log2);

    const float inv_sum[2] = {1.0f / quad_sum(row_sum[0]), 1.0f / quad_sum(row_sum[1])};
#pragma unroll
    for (int block = 0; block < D_BLOCKS; ++block) {
        out[block][0] *= inv_sum[0];
        out[block][1] *= inv_sum[0];
        out[block][2] *= inv_sum[1];
        out[block][3] *= inv_sum[1];
        store_acc(O + block * MMA_N, HEAD_DIM, out[block], row, col);
    }
}

constexpr int WARPS_PER_BLOCK = UNICA_WARPS_PER_BLOCK;
constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * ROWS_PER_WARP;  // 64 query rows per block

// Step 5: the whole attention, any lengths. Q and O are [heads][n_queries][HEAD_DIM], K and V are
// [heads][n_keys][HEAD_DIM], all bf16 and contiguous; "heads" is batch · heads.
// Grid: (ceil(n_queries / ROWS_PER_BLOCK), heads). Block: WARPS_PER_BLOCK warps, ROWS_PER_WARP rows each.
// Every warp reads all of K and V of its head on its own: no shared memory yet (M2).
extern "C" __global__ void attention(const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
                                     __nv_bfloat16* O, int n_queries, int n_keys, float scale_log2) {
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int row = lane / 4;
    const int col = 2 * (lane % 4);

    const int row0 = blockIdx.x * ROWS_PER_BLOCK + warp * ROWS_PER_WARP;
    const int n_rows = min(n_queries - row0, ROWS_PER_WARP);
    if (n_rows <= 0) return;  // the whole warp at once: no lane is left waiting in a shuffle

    // size_t: 40 heads × 32760 × 128 still fits in int, but batch 16 of them does not
    const size_t q_offset = (size_t(blockIdx.y) * n_queries + row0) * HEAD_DIM;
    const size_t kv_offset = size_t(blockIdx.y) * n_keys * HEAD_DIM;
    K += kv_offset;
    V += kv_offset;

    uint32_t q[Q_STEPS][4];
    load_q(q, Q + q_offset, row, col, n_rows);

    float row_max[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float row_sum[2] = {0.0f, 0.0f};
    float out[D_BLOCKS][4] = {};

    const int full_keys = n_keys - n_keys % KEYS_PER_TILE;
    for (int key0 = 0; key0 < full_keys; key0 += KEYS_PER_TILE)
        attention_tile<false>(q, K + key0 * HEAD_DIM, V + key0 * HEAD_DIM, KEYS_PER_TILE, row_max, row_sum, out,
                              row, col, scale_log2);
    if (full_keys < n_keys)
        attention_tile<true>(q, K + full_keys * HEAD_DIM, V + full_keys * HEAD_DIM, n_keys - full_keys, row_max,
                             row_sum, out, row, col, scale_log2);

    store_out(O + q_offset, out, row_sum, row, col, n_rows);
}

// ---------------------------------------------------------------------------------------------------
// M2: speed. Each step is a separate kernel next to the one before it, so bench/compare.py can time
// both in one run.
// ---------------------------------------------------------------------------------------------------

constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
constexpr int CHUNK = 8;                          // bf16 in one 16-byte load
constexpr int CHUNKS_PER_ROW = HEAD_DIM / CHUNK;  // 16
constexpr int TILE_LOADS = KEYS_PER_TILE * CHUNKS_PER_ROW / THREADS_PER_BLOCK;  // 8 per thread
static_assert(KEYS_PER_TILE * CHUNKS_PER_ROW % THREADS_PER_BLOCK == 0, "a tile must split evenly over the block");

// Rows in shared memory are 8 bf16 longer than in K and V. Shared memory has 32 banks of 4 bytes, and a
// 128-wide row is 256 bytes, exactly two rounds of banks: the 8 rows one ldmatrix reads would all start
// in bank 0 and queue up 8 deep. 272 bytes per row moves row i to bank 4·i: 0, 4 .. 28, no queue.
constexpr int SMEM_STRIDE = HEAD_DIM + 8;

// KEYS_PER_TILE rows of K or V into shared memory, all threads of the block together, 16 bytes per load.
// Neighbouring threads take neighbouring chunks of a row, so each warp reads 512 contiguous bytes.
// Rows at or past n_keys become zeros (see v_at for why zeros and not just "skip").
__device__ __forceinline__ void load_tile(__nv_bfloat16* tile, const __nv_bfloat16* src, int n_keys) {
#pragma unroll
    for (int load = 0; load < TILE_LOADS; ++load) {
        const int chunk_index = load * THREADS_PER_BLOCK + threadIdx.x;
        const int key = chunk_index / CHUNKS_PER_ROW;
        const int dim = chunk_index % CHUNKS_PER_ROW * CHUNK;
        uint4 data = make_uint4(0, 0, 0, 0);
        if (key < n_keys) data = *reinterpret_cast<const uint4*>(src + key * HEAD_DIM + dim);
        *reinterpret_cast<uint4*>(tile + key * SMEM_STRIDE + dim) = data;
    }
}

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

// The same, transposed on the way in: the lane gets rows 2 · (lane % 4) and + 1 of column lane / 4.
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t r[4], const void* lane_row) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_address(lane_row)));
}

__device__ __forceinline__ uint32_t bf16x2_to_f16x2(uint32_t pair) {
    const float2 values = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&pair));
    const __half2 halves = __float22half2_rn(values);
    return *reinterpret_cast<const uint32_t*>(&halves);
}

// scores = Q · Kᵀ with the K tile in shared memory. One ldmatrix.x4 brings the B fragments of two
// neighbouring key blocks: matrices 0, 1 are keys 0..7 of the pair, dims 0..7 and 8..15 (b[0], b[1] of
// the first block); matrices 2, 3 are keys 8..15, the same dims (the second block).
// 32 ldmatrix per tile instead of 128 4-byte loads.
__device__ __forceinline__ void qk_scores_smem(float scores[KEY_BLOCKS][4], const uint32_t q[Q_STEPS][4],
                                               const __nv_bfloat16* k_tile, int lane) {
    const int matrix = lane / 8, matrix_row = lane % 8;
    const __nv_bfloat16* lane_row = k_tile + (matrix / 2 * MMA_N + matrix_row) * SMEM_STRIDE + matrix % 2 * 8;
#pragma unroll
    for (int block = 0; block < KEY_BLOCKS; ++block)
#pragma unroll
        for (int i = 0; i < 4; ++i) scores[block][i] = 0.0f;
#pragma unroll
    for (int step = 0; step < Q_STEPS; ++step)
#pragma unroll
        for (int pair = 0; pair < KEY_BLOCKS / 2; ++pair) {
            uint32_t b[4];
            ldmatrix_x4(b, lane_row + pair * 2 * MMA_N * SMEM_STRIDE + step * MMA_K);
            mma_bf16_16816(scores[2 * pair], q[step], b);
            mma_bf16_16816(scores[2 * pair + 1], q[step], b + 2);
        }
}

// out += P · V with the V tile in shared memory. V is stored [key][dim] but the B fragment wants two
// neighbouring keys of one dim: ldmatrix.trans turns it on the way in. Matrices 0, 1 are keys 0..7 and
// 8..15 of the step for one block of 8 dims (b[0], b[1]); matrices 2, 3 the same keys for the next block.
// 32 ldmatrix per tile instead of 256 2-byte loads.
__device__ __forceinline__ void pv_smem(float out[D_BLOCKS][4], const float scores[KEY_BLOCKS][4],
                                        const __nv_bfloat16* v_tile, int lane) {
    const int matrix = lane / 8, matrix_row = lane % 8;
    const __nv_bfloat16* lane_row = v_tile + (matrix % 2 * 8 + matrix_row) * SMEM_STRIDE + matrix / 2 * MMA_N;
#pragma unroll
    for (int step = 0; step < KEY_STEPS; ++step) {
        uint32_t p[4];
        p_fragment(p, scores, step);
#pragma unroll
        for (int pair = 0; pair < D_BLOCKS / 2; ++pair) {
            uint32_t b[4];
            ldmatrix_x4_trans(b, lane_row + step * MMA_K * SMEM_STRIDE + pair * 2 * MMA_N);
#pragma unroll
            for (int i = 0; i < 4; ++i) b[i] = bf16x2_to_f16x2(b[i]);
            mma_f16_16816(out[2 * pair], p, b);
            mma_f16_16816(out[2 * pair + 1], p, b + 2);
        }
    }
}

// M2 step 1: K and V tiles go through shared memory once per block, warps take fragments with ldmatrix.
// Same grid, block and arguments as attention. MIN_BLOCKS_PER_SM > 1 makes ptxas fit the registers so
// that many blocks share an SM, spilling if it must.
extern "C" __global__ void __launch_bounds__(THREADS_PER_BLOCK, UNICA_MIN_BLOCKS_PER_SM)
    attention_smem(const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V, __nv_bfloat16* O,
                   int n_queries, int n_keys, float scale_log2) {
    __shared__ __align__(16) __nv_bfloat16 k_tile[KEYS_PER_TILE * SMEM_STRIDE];
    __shared__ __align__(16) __nv_bfloat16 v_tile[KEYS_PER_TILE * SMEM_STRIDE];

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int row = lane / 4;
    const int col = 2 * (lane % 4);

    // A warp past the last query row does not return: it still loads its share of every tile and meets
    // every __syncthreads. Its Q reads as zero and it writes nothing.
    const int row0 = blockIdx.x * ROWS_PER_BLOCK + warp * ROWS_PER_WARP;
    const int n_rows = min(n_queries - row0, ROWS_PER_WARP);

    const size_t q_offset = (size_t(blockIdx.y) * n_queries + row0) * HEAD_DIM;
    const size_t kv_offset = size_t(blockIdx.y) * n_keys * HEAD_DIM;
    K += kv_offset;
    V += kv_offset;

    uint32_t q[Q_STEPS][4];
    load_q(q, Q + q_offset, row, col, n_rows);

    float row_max[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float row_sum[2] = {0.0f, 0.0f};
    float out[D_BLOCKS][4] = {};

    for (int key0 = 0; key0 < n_keys; key0 += KEYS_PER_TILE) {
        const int n_valid = min(n_keys - key0, KEYS_PER_TILE);
        __syncthreads();  // every warp is done reading the previous tile
        load_tile(k_tile, K + key0 * HEAD_DIM, n_valid);
        load_tile(v_tile, V + key0 * HEAD_DIM, n_valid);
        __syncthreads();  // the whole tile is on the board

        float scores[KEY_BLOCKS][4], alpha[2];
        qk_scores_smem(scores, q, k_tile, lane);
        if (n_valid < KEYS_PER_TILE) mask_scores(scores, col, n_valid);
        online_softmax(scores, row_max, row_sum, alpha, scale_log2);
        rescale_out(out, alpha);
        pv_smem(out, scores, v_tile, lane);
    }

    store_out(O + q_offset, out, row_sum, row, col, n_rows);
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
constexpr int K8_TILE_LOADS = KEYS_PER_TILE * HEAD_DIM / 16 / THREADS_PER_BLOCK;  // 16-byte loads per thread
static_assert(KEYS_PER_TILE * HEAD_DIM / 16 % THREADS_PER_BLOCK == 0, "an int8 tile must split evenly");

// 16 rows of int8 Q as A fragments; rows at or past n_rows read as zero.
__device__ __forceinline__ void load_q8(uint32_t q[Q8_STEPS][4], const int8_t* Q, int row, int col, int n_rows) {
    const bool top = row < n_rows, bottom = row + 8 < n_rows;
    const int byte = 2 * col;  // 4 int8 per register
    auto word = [](const int8_t* ptr) { return *reinterpret_cast<const uint32_t*>(ptr); };
#pragma unroll
    for (int step = 0; step < Q8_STEPS; ++step) {
        const int8_t* tile = Q + step * MMA_K8 + byte;
        q[step][0] = top ? word(tile + row * HEAD_DIM) : 0;
        q[step][1] = bottom ? word(tile + (row + 8) * HEAD_DIM) : 0;
        q[step][2] = top ? word(tile + row * HEAD_DIM + 16) : 0;
        q[step][3] = bottom ? word(tile + (row + 8) * HEAD_DIM + 16) : 0;
    }
}

// Where key k of a group of 16 sits in the V tile of attention_int8pv. P leaves Q·Kᵀ with keys 2t, 2t+1,
// 8+2t, 9+2t in lane t's four bytes (t = lane % 4), and m16n8k32 reads those bytes as k = 4t .. 4t+3. The
// sum over k does not care about the order, only that P and V agree: V's rows are stored in P's order,
// slot 4t + j holds key 2t + j % 2 + 8 · (j / 2). Keys 0, 1, 8, 9, 2, 3, 10, 11 .. go to slots 0, 1, 2, 3 ..
__device__ __forceinline__ int p_order_slot(int key) {
    const int k = key % 16;
    return key - k + 2 * (k % 8) - k % 2 + 2 * (k / 8);
}

// KEYS_PER_TILE rows of int8 K into shared memory, as load_tile does for bf16. P_ORDER: rows go to
// p_order_slot(key), for V in attention_int8pv.
template <bool P_ORDER = false>
__device__ __forceinline__ void load_tile8(int8_t* tile, const int8_t* src, int n_keys) {
#pragma unroll
    for (int load = 0; load < K8_TILE_LOADS; ++load) {
        const int chunk_index = load * THREADS_PER_BLOCK + threadIdx.x;
        const int key = chunk_index / (HEAD_DIM / 16);
        const int byte = chunk_index % (HEAD_DIM / 16) * 16;
        uint4 data = make_uint4(0, 0, 0, 0);
        if (key < n_keys) data = *reinterpret_cast<const uint4*>(src + key * HEAD_DIM + byte);
        *reinterpret_cast<uint4*>(tile + (P_ORDER ? p_order_slot(key) : key) * K8_STRIDE + byte) = data;
    }
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

// M3 step 1: Q·Kᵀ in int8, P·V as in attention_smem. Q8 and K8 are int8 [heads][tokens][HEAD_DIM]
// with K smoothed (its mean over tokens subtracted: softmax cancels it); q_scale and k_scale hold one
// scale per Q_SCALE_BLOCK and K_SCALE_BLOCK tokens, [heads][blocks]. q·k ≈ q̂·k̂ · q_scale · k_scale.
// Same grid and block as attention_smem.
extern "C" __global__ void __launch_bounds__(THREADS_PER_BLOCK, UNICA_MIN_BLOCKS_PER_SM)
    attention_int8qk(const int8_t* Q8, const int8_t* K8, const float* q_scale, const float* k_scale,
                     const __nv_bfloat16* V, __nv_bfloat16* O, int n_queries, int n_keys, float scale_log2) {
    __shared__ __align__(16) int8_t k_tile[KEYS_PER_TILE * K8_STRIDE];
    __shared__ __align__(16) __nv_bfloat16 v_tile[KEYS_PER_TILE * SMEM_STRIDE];

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int row = lane / 4;
    const int col = 2 * (lane % 4);

    const int row0 = blockIdx.x * ROWS_PER_BLOCK + warp * ROWS_PER_WARP;
    const int n_rows = min(n_queries - row0, ROWS_PER_WARP);

    const size_t q_offset = (size_t(blockIdx.y) * n_queries + row0) * HEAD_DIM;
    const size_t kv_offset = size_t(blockIdx.y) * n_keys * HEAD_DIM;
    K8 += kv_offset;
    V += kv_offset;
    k_scale += size_t(blockIdx.y) * ((n_keys + K_SCALE_BLOCK - 1) / K_SCALE_BLOCK);
    const float q_scale_log2 =
        q_scale[size_t(blockIdx.y) * ((n_queries + Q_SCALE_BLOCK - 1) / Q_SCALE_BLOCK) + row0 / Q_SCALE_BLOCK] *
        scale_log2;

    uint32_t q[Q8_STEPS][4];
    load_q8(q, Q8 + q_offset, row, col, n_rows);

    float row_max[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float row_sum[2] = {0.0f, 0.0f};
    float out[D_BLOCKS][4] = {};

    for (int key0 = 0; key0 < n_keys; key0 += KEYS_PER_TILE) {
        const int n_valid = min(n_keys - key0, KEYS_PER_TILE);
        __syncthreads();
        load_tile8(k_tile, K8 + key0 * HEAD_DIM, n_valid);
        load_tile(v_tile, V + key0 * HEAD_DIM, n_valid);
        __syncthreads();

        float scores[KEY_BLOCKS][4], alpha[2];
        qk_scores_int8(scores, q, k_tile, lane);
        if (n_valid < KEYS_PER_TILE) mask_scores(scores, col, n_valid);
        // The whole tile shares one Q scale and one K scale, so dequantization is one number in the place
        // of scale_log2: no instruction of its own.
        online_softmax(scores, row_max, row_sum, alpha, q_scale_log2 * k_scale[key0 / K_SCALE_BLOCK]);
        rescale_out(out, alpha);
        pv_smem(out, scores, v_tile, lane);
    }

    store_out(O + q_offset, out, row_sum, row, col, n_rows);
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
// every x ≥ −126 (exp2_ftz_mismatches checks all of them).
__device__ __forceinline__ float exp2_ftz(float x) {
#if UNICA_FAST_EXP2
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
#else
    return exp2f(x);
#endif
}

// Every float bit pattern goes through exp2f and exp2_ftz; count[0] gets the inputs ≥ −126 where they differ
// (expected 0), count[1] those below −126 (or NaN) where they differ. Grid of any size: the threads stride
// over all 2^32 patterns.
extern "C" __global__ void exp2_ftz_mismatches(unsigned* count) {
    unsigned in_range = 0, below = 0;
    const unsigned stride = gridDim.x * blockDim.x;
    for (unsigned bits = blockIdx.x * blockDim.x + threadIdx.x, done = 0; !done; bits += stride) {
        const float x = __uint_as_float(bits);
        if (__float_as_uint(exp2_ftz(x)) != __float_as_uint(exp2f(x))) ++(x >= -126.0f ? in_range : below);
        done = bits > 0xffffffffu - stride;
    }
    atomicAdd(&count[0], in_range);
    atomicAdd(&count[1], below);
}

// online_softmax for a u8 P. Each row's P is taken against the max of this tile, not the running max, and
// comes out as 255 · 2^(s · scale_log2 − tile max), in [0, 255]: the largest p of every row and tile is 255,
// so a tile under an older, larger max still gets all 256 steps. beta = 2^(tile max − running max) brings it
// back when P·V is added into out. row_sum gathers the unrounded values: it is 255 times the usual sum, P·V
// carries the same 255, and the two cancel in out / sum. With UNICA_SUM_ROUNDED row_sum is left alone here:
// add_row_sum_u8 adds the rounded P̂ after packing.
__device__ __forceinline__ void online_softmax_u8(float scores[KEY_BLOCKS][4], float row_max[2], float row_sum[2],
                                                  float alpha[2], float beta[2], float scale_log2) {
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
        [[maybe_unused]] float tile_sum = 0.0f;
#pragma unroll
        for (int block = 0; block < KEY_BLOCKS; ++block)
#pragma unroll
            for (int i = 2 * half; i < 2 * half + 2; ++i) {
                scores[block][i] = exp2_ftz(fmaf(scores[block][i], scale_log2, offset));
                if (!UNICA_SUM_ROUNDED) tile_sum += scores[block][i];
            }
        if (!UNICA_SUM_ROUNDED) row_sum[half] = row_sum[half] * alpha[half] + tile_sum * beta[half];
        row_max[half] = new_max;
    }
}

// M4.44: a skipped tile (UNICA_SKIP_TILES) leaves the denominator as it is, which needs it gathered by add_row_sum_u8
// after packing, not by online_softmax_u8 before the skip is decided.
static_assert(!(UNICA_SKIP_TILES || UNICA_SKIP_BLOCKS) || UNICA_SUM_ROUNDED, "SKIP_* needs SUM_ROUNDED");

// M4.44 H5: the tile max of both rows, scaled to log2 units, as online_softmax_u8 takes it.
__device__ __forceinline__ void tile_max_log2(const float scores[KEY_BLOCKS][4], float tile_max[2], float scale_log2) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        tile_max[half] = -CUDART_INF_F;
#pragma unroll
        for (int block = 0; block < KEY_BLOCKS; ++block)
            tile_max[half] = fmaxf(tile_max[half], fmaxf(scores[block][2 * half], scores[block][2 * half + 1]));
        tile_max[half] = quad_max(tile_max[half]) * scale_log2;
    }
}

// M4.44 H5: online_softmax_u8 (SUM_ROUNDED) with the tile max taken a tile earlier, by tile_max_log2. A copy, so that
// the default build keeps its SASS.
__device__ __forceinline__ void online_softmax_u8_premax(float scores[KEY_BLOCKS][4], const float tile_max[2],
                                                         float row_max[2], float alpha[2], float beta[2], float scale_log2) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const float new_max = fmaxf(row_max[half], tile_max[half]);
        alpha[half] = exp2_ftz(row_max[half] - new_max);
        beta[half] = exp2_ftz(tile_max[half] - new_max);
        const float offset = LOG2_255 - tile_max[half];
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
    if (!UNICA_SUM_ROUNDED) return;
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
// exact in int32, and |sum| ≤ 64 · 255 · 127 < 2^24 keeps even the conversion to float exact. The loops are
// turned around from pv_smem: one pair of d blocks takes both 32-key steps and is added into out at once,
// so 8 int32 accumulators are alive at a time, not 64. PAIRED (M3.17): V̂ in the paired rows of v8_offset's
// layout 2, V8_STRIDE apart, whose ldmatrix returns each B operand in adjacent registers: no MOV to pair them.
constexpr int V8_STRIDE = 2 * HEAD_DIM + 16;  // a paired row and 16 bytes of padding: row i starts in bank 4 · i
static_assert(KEYS_PER_TILE / 2 * V8_STRIDE <= KEYS_PER_TILE * K8_STRIDE, "a paired V tile must fit a V tile");

template <bool PAIRED = false>
__device__ __forceinline__ void pv_int8(float out[D_BLOCKS][4], const uint32_t p[PV8_STEPS][4], const int8_t* v_tile,
                                        const float alpha[2], const float beta[2], int lane) {
#if UNICA_LAZY_RESCALE
    // M3.9: alpha is exactly 1 unless the max of a row grew in this tile, and then out · alpha is out. When it
    // is 1 in every row of the warp, the 64 multiplies are skipped. Rounding is the same on both paths.
    if (!__all_sync(0xffffffff, alpha[0] == 1.0f && alpha[1] == 1.0f)) rescale_out(out, alpha);
#endif
    // unpaired: keys 0..15 of a step for matrix 0, 16..31 for matrix 1. Paired: row j of the step for both,
    // matrix 0 reading the left block of dims of the pair, matrix 1 the right one.
    const int8_t* lane_row = PAIRED ? v_tile + lane % 16 * V8_STRIDE + lane / 16 * 16 : v_tile + lane * K8_STRIDE;
#pragma unroll
    for (int pair = 0; pair < D_BLOCKS / 2; ++pair) {
        int acc[2][4] = {};
#pragma unroll
        for (int step = 0; step < PV8_STEPS; ++step) {
            uint32_t b[4];
            if (PAIRED) {
                ldmatrix_x2_trans_b8(b, lane_row + step * 16 * V8_STRIDE + pair * 32);
            } else {
                ldmatrix_x2_trans_b8(b, lane_row + step * MMA_K8 * K8_STRIDE + pair * 2 * MMA_N);
            }
            const uint32_t left[2] = {b[0], PAIRED ? b[1] : b[2]}, right[2] = {PAIRED ? b[2] : b[1], b[3]};
            mma_u8s8_16832(acc[0], p[step], left);
            mma_u8s8_16832(acc[1], p[step], right);
        }
#pragma unroll
        for (int half = 0; half < 2; ++half)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float& o = out[2 * pair + half][i];
                o = fmaf(beta[i / 2], __int2float_rn(acc[half][i]), UNICA_LAZY_RESCALE ? o : o * alpha[i / 2]);
            }
    }
}

// M3.3 step 1: P (16 × 64 float, values in [0, 255]) times V (64 × HEAD_DIM int8) through the path of
// attention_int8pv: rounding, packing, the slot order, the 8-bit ldmatrix, the u8 × s8 mma. O is 16 × HEAD_DIM
// float, exact. Launch with THREADS_PER_BLOCK threads: all of them load the tile, warp 0 computes.
extern "C" __global__ void pv_int8_tile(const float* P, const int8_t* V, float* O) {
    __shared__ __align__(16) int8_t v_tile[KEYS_PER_TILE * K8_STRIDE];
    load_tile8<true>(v_tile, V, KEYS_PER_TILE);
    __syncthreads();
    if (threadIdx.x >= WARP_SIZE) return;

    const int lane = threadIdx.x;
    const int row = lane / 4;
    const int col = 2 * (lane % 4);
    float scores[KEY_BLOCKS][4];
#pragma unroll
    for (int block = 0; block < KEY_BLOCKS; ++block)
#pragma unroll
        for (int i = 0; i < 4; ++i)
            scores[block][i] = P[(row + i / 2 * 8) * KEYS_PER_TILE + block * MMA_N + col + i % 2];
    uint32_t p[PV8_STEPS][4];
#pragma unroll
    for (int step = 0; step < PV8_STEPS; ++step) p_fragment_u8(p[step], scores, step);

    float out[D_BLOCKS][4] = {};
    const float one[2] = {1.0f, 1.0f};
    pv_int8(out, p, v_tile, one, one, lane);
#pragma unroll
    for (int block = 0; block < D_BLOCKS; ++block) store_acc(O + block * MMA_N, HEAD_DIM, out[block], row, col);
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

// M3.3 step 2: attention_int8qk with P·V in u8 × s8. V8 is int8 [heads][n_keys][HEAD_DIM], v_scale float
// [heads][HEAD_DIM]. Same grid and block as attention_smem.
extern "C" __global__ void __launch_bounds__(THREADS_PER_BLOCK, UNICA_MIN_BLOCKS_PER_SM)
    attention_int8pv(const int8_t* Q8, const int8_t* K8, const float* q_scale, const float* k_scale,
                     const int8_t* V8, const float* v_scale, __nv_bfloat16* O, int n_queries, int n_keys,
                     float scale_log2) {
    __shared__ __align__(16) int8_t k_tile[KEYS_PER_TILE * K8_STRIDE];
    __shared__ __align__(16) int8_t v_tile[KEYS_PER_TILE * K8_STRIDE];

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int row = lane / 4;
    const int col = 2 * (lane % 4);

    const int row0 = blockIdx.x * ROWS_PER_BLOCK + warp * ROWS_PER_WARP;
    const int n_rows = min(n_queries - row0, ROWS_PER_WARP);

    const size_t q_offset = (size_t(blockIdx.y) * n_queries + row0) * HEAD_DIM;
    const size_t kv_offset = size_t(blockIdx.y) * n_keys * HEAD_DIM;
    K8 += kv_offset;
    V8 += kv_offset;
    v_scale += size_t(blockIdx.y) * HEAD_DIM;
    k_scale += size_t(blockIdx.y) * ((n_keys + K_SCALE_BLOCK - 1) / K_SCALE_BLOCK);
    const float q_scale_log2 =
        q_scale[size_t(blockIdx.y) * ((n_queries + Q_SCALE_BLOCK - 1) / Q_SCALE_BLOCK) + row0 / Q_SCALE_BLOCK] *
        scale_log2;

    uint32_t q[Q8_STEPS][4];
    load_q8(q, Q8 + q_offset, row, col, n_rows);

    float row_max[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float row_sum[2] = {0.0f, 0.0f};
    float out[D_BLOCKS][4] = {};

    for (int key0 = 0; key0 < n_keys; key0 += KEYS_PER_TILE) {
        const int n_valid = min(n_keys - key0, KEYS_PER_TILE);
        __syncthreads();
        load_tile8(k_tile, K8 + key0 * HEAD_DIM, n_valid);
        load_tile8<true>(v_tile, V8 + key0 * HEAD_DIM, n_valid);
        __syncthreads();

        float scores[KEY_BLOCKS][4], alpha[2], beta[2];
        qk_scores_int8(scores, q, k_tile, lane);
        if (n_valid < KEYS_PER_TILE) mask_scores(scores, col, n_valid);
        online_softmax_u8(scores, row_max, row_sum, alpha, beta, q_scale_log2 * k_scale[key0 / K_SCALE_BLOCK]);
        uint32_t p[PV8_STEPS][4];
#pragma unroll
        for (int step = 0; step < PV8_STEPS; ++step) p_fragment_u8(p[step], scores, step);
        add_row_sum_u8(row_sum, p, alpha, beta);
        pv_int8(out, p, v_tile, alpha, beta, lane);
    }

    store_out(O + q_offset, out, row_sum, row, col, n_rows, v_scale);
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

// load_tile8 with cp.async: the same chunks and slots, the copies only start here. Keys at or past n_keys
// become zeros, read from nowhere: the address stays at key 0 of the tile, which always exists.
template <bool P_ORDER = false>
__device__ __forceinline__ void load_tile8_async(int8_t* tile, const int8_t* src, int n_keys) {
#pragma unroll
    for (int load = 0; load < K8_TILE_LOADS; ++load) {
        const int chunk_index = load * THREADS_PER_BLOCK + threadIdx.x;
        const int key = chunk_index / (HEAD_DIM / 16);
        const int byte = chunk_index % (HEAD_DIM / 16) * 16;
        const bool valid = key < n_keys;
        cp_async16(tile + (P_ORDER ? p_order_slot(key) : key) * K8_STRIDE + byte,
                   src + (valid ? key : 0) * HEAD_DIM + byte, valid ? 16 : 0);
    }
}

// M3.8: a timeline of the tile loop for the warps of SM 0, built only with -DUNICA_TRACE=1 (bench/timeline.py).
// Each such warp takes a slot and records clock64 at TRACE_POINTS places of every tile; trace_read copies it
// all out. Without UNICA_TRACE the marks are empty and the kernel is the same instruction for instruction.
#ifndef UNICA_TRACE
#define UNICA_TRACE 0
#endif
#if UNICA_TRACE
constexpr int TRACE_SLOTS = 64, TRACE_TILES = 1024, TRACE_POINTS = 4;  // bench/timeline.py has them too
__device__ long long trace_clocks[TRACE_SLOTS][TRACE_TILES][TRACE_POINTS];
__device__ int trace_warp[TRACE_SLOTS];  // %warpid: the scheduler is %warpid % 4
__device__ int trace_used;

// The slot of this warp, or -1 when it is not on SM 0 or the slots are full. Same value in every lane.
__device__ __forceinline__ int trace_slot(int lane) {
    unsigned sm, warp_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    asm volatile("mov.u32 %0, %%warpid;" : "=r"(warp_id));
    int slot = -1;
    if (sm == 0 && lane == 0) {
        slot = atomicAdd(&trace_used, 1);
        if (slot < TRACE_SLOTS) trace_warp[slot] = warp_id;
        else slot = -1;
    }
    return __shfl_sync(0xffffffff, slot, 0);
}

__device__ __forceinline__ void trace_mark(int slot, int lane, int tile, int point) {
    long long now;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(now) :: "memory");
    if (slot >= 0 && lane == 0 && tile < TRACE_TILES) trace_clocks[slot][tile][point] = now;
}

// The same, read only once `after` is computed: the value is an input of the clock read, so the read waits for it.
// This marks when a chain of work ends, wherever ptxas put the instructions of that chain (M4.10).
__device__ __forceinline__ void trace_mark_after(int slot, int lane, int tile, int point, uint32_t after) {
    long long now;
    asm volatile("{ .reg .u32 unused; mov.u32 unused, %1; mov.u64 %0, %%clock64; }" : "=l"(now) : "r"(after) : "memory");
    if (slot >= 0 && lane == 0 && tile < TRACE_TILES) trace_clocks[slot][tile][point] = now;
}

// clocks: int64 [TRACE_SLOTS][TRACE_TILES][TRACE_POINTS], warps: int32 [TRACE_SLOTS], used: int32 [1]. Then
// clears the slots for the next run. One thread.
extern "C" __global__ void trace_read(long long* clocks, int* warps, int* used) {
    for (int i = 0; i < TRACE_SLOTS * TRACE_TILES * TRACE_POINTS; ++i) clocks[i] = (&trace_clocks[0][0][0])[i];
    for (int i = 0; i < TRACE_SLOTS; ++i) warps[i] = trace_warp[i];
    *used = min(trace_used, TRACE_SLOTS);
    trace_used = 0;
}
#define TRACE_SLOT(lane) trace_slot(lane)
#define TRACE_MARK(slot, lane, tile, point) trace_mark(slot, lane, tile, point)
#define TRACE_MARK_AFTER(slot, lane, tile, point, after) trace_mark_after(slot, lane, tile, point, after)
#else
#define TRACE_SLOT(lane) (-1)
#define TRACE_MARK(slot, lane, tile, point) ((void)0)
#define TRACE_MARK_AFTER(slot, lane, tile, point, after) ((void)0)
#endif

// M3.4: attention_int8pv with two buffers per tile. Tile t + 1 is on its way while tile t is computed, and
// one __syncthreads per tile does the work of two: at the top of tile t it means both "tile t has landed"
// and "every warp is done with tile t − 1", whose buffer the load of tile t + 1 then reuses.
// Same arguments, grid and block as attention_int8pv.
extern "C" __global__ void __launch_bounds__(THREADS_PER_BLOCK, UNICA_MIN_BLOCKS_PER_SM)
    attention_int8pv_async(const int8_t* Q8, const int8_t* K8, const float* q_scale, const float* k_scale,
                           const int8_t* V8, const float* v_scale, __nv_bfloat16* O, int n_queries, int n_keys,
                           float scale_log2) {
    __shared__ __align__(16) int8_t k_tiles[2][KEYS_PER_TILE * K8_STRIDE];
    __shared__ __align__(16) int8_t v_tiles[2][KEYS_PER_TILE * K8_STRIDE];

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int row = lane / 4;
    const int col = 2 * (lane % 4);

    const int row0 = blockIdx.x * ROWS_PER_BLOCK + warp * ROWS_PER_WARP;
    const int n_rows = min(n_queries - row0, ROWS_PER_WARP);

    const size_t q_offset = (size_t(blockIdx.y) * n_queries + row0) * HEAD_DIM;
    const size_t kv_offset = size_t(blockIdx.y) * n_keys * HEAD_DIM;
    K8 += kv_offset;
    V8 += kv_offset;
    v_scale += size_t(blockIdx.y) * HEAD_DIM;
    k_scale += size_t(blockIdx.y) * ((n_keys + K_SCALE_BLOCK - 1) / K_SCALE_BLOCK);
    const float q_scale_log2 =
        q_scale[size_t(blockIdx.y) * ((n_queries + Q_SCALE_BLOCK - 1) / Q_SCALE_BLOCK) + row0 / Q_SCALE_BLOCK] *
        scale_log2;

    load_tile8_async(k_tiles[0], K8, min(n_keys, KEYS_PER_TILE));
    load_tile8_async<true>(v_tiles[0], V8, min(n_keys, KEYS_PER_TILE));
    cp_async_commit();

    uint32_t q[Q8_STEPS][4];
    load_q8(q, Q8 + q_offset, row, col, n_rows);

    float row_max[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float row_sum[2] = {0.0f, 0.0f};
    float out[D_BLOCKS][4] = {};
    [[maybe_unused]] const int slot = TRACE_SLOT(lane);

    for (int key0 = 0, buffer = 0; key0 < n_keys; key0 += KEYS_PER_TILE, buffer ^= 1) {
        cp_async_wait();  // this thread's part of tile key0
        __syncthreads();  // everyone's part; and the other buffer is free
        const int next = key0 + KEYS_PER_TILE;
        if (next < n_keys) {
            load_tile8_async(k_tiles[buffer ^ 1], K8 + next * HEAD_DIM, min(n_keys - next, KEYS_PER_TILE));
            load_tile8_async<true>(v_tiles[buffer ^ 1], V8 + next * HEAD_DIM, min(n_keys - next, KEYS_PER_TILE));
            cp_async_commit();
        }

        [[maybe_unused]] const int tile = key0 / KEYS_PER_TILE;
        TRACE_MARK(slot, lane, tile, 0);
        const int n_valid = min(n_keys - key0, KEYS_PER_TILE);
        float scores[KEY_BLOCKS][4], alpha[2], beta[2];
        qk_scores_int8(scores, q, k_tiles[buffer], lane);
        TRACE_MARK(slot, lane, tile, 1);
        if (n_valid < KEYS_PER_TILE) mask_scores(scores, col, n_valid);
        online_softmax_u8(scores, row_max, row_sum, alpha, beta, q_scale_log2 * k_scale[key0 / K_SCALE_BLOCK]);
        uint32_t p[PV8_STEPS][4];
#pragma unroll
        for (int step = 0; step < PV8_STEPS; ++step) p_fragment_u8(p[step], scores, step);
        add_row_sum_u8(row_sum, p, alpha, beta);
        TRACE_MARK(slot, lane, tile, 2);
        pv_int8(out, p, v_tiles[buffer], alpha, beta, lane);
        TRACE_MARK(slot, lane, tile, 3);
    }

    store_out(O + q_offset, out, row_sum, row, col, n_rows, v_scale);
}

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

// attention_int8pv_async with Q·Kᵀ(t + 1) issued before the softmax of tile t. The two are independent, so
// ptxas can put the softmax between the IMMA, and the tensor pipe has work while the softmax runs. No branch
// may stand between them, or they land in different basic blocks: the mask of a partial tile goes on its
// scores right after they are made, at the end of the iteration before, and Q·Kᵀ(t + 1) runs even in the
// last iteration, on stale K, and is thrown away. Iteration t reads K(t + 1) and V(t), and loads K(t + 2)
// into the K buffer of tile t (Q·Kᵀ(t) read it in the iteration before) and V(t + 1) into the other V buffer.
// Same arguments, grid and block as attention_int8pv.
extern "C" __global__ void __launch_bounds__(THREADS_PER_BLOCK, UNICA_MIN_BLOCKS_PER_SM)
    attention_int8pv_pipe(const int8_t* Q8, const int8_t* K8, const float* q_scale, const float* k_scale,
                          const int8_t* V8, const float* v_scale, __nv_bfloat16* O, int n_queries, int n_keys,
                          float scale_log2) {
    __shared__ __align__(16) int8_t k_tiles[2][KEYS_PER_TILE * K8_STRIDE];
    __shared__ __align__(16) int8_t v_tiles[2][KEYS_PER_TILE * K8_STRIDE];

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int row = lane / 4;
    const int col = 2 * (lane % 4);

    const int row0 = blockIdx.x * ROWS_PER_BLOCK + warp * ROWS_PER_WARP;
    const int n_rows = min(n_queries - row0, ROWS_PER_WARP);

    const size_t q_offset = (size_t(blockIdx.y) * n_queries + row0) * HEAD_DIM;
    const size_t kv_offset = size_t(blockIdx.y) * n_keys * HEAD_DIM;
    K8 += kv_offset;
    V8 += kv_offset;
    v_scale += size_t(blockIdx.y) * HEAD_DIM;
    k_scale += size_t(blockIdx.y) * ((n_keys + K_SCALE_BLOCK - 1) / K_SCALE_BLOCK);
    const float q_scale_log2 =
        q_scale[size_t(blockIdx.y) * ((n_queries + Q_SCALE_BLOCK - 1) / Q_SCALE_BLOCK) + row0 / Q_SCALE_BLOCK] *
        scale_log2;
    const int n_tiles = (n_keys + KEYS_PER_TILE - 1) / KEYS_PER_TILE;

    load_tile8_async(k_tiles[0], K8, min(n_keys, KEYS_PER_TILE));
    load_tile8_async<true>(v_tiles[0], V8, min(n_keys, KEYS_PER_TILE));
    if (n_tiles > 1)
        load_tile8_async(k_tiles[1], K8 + KEYS_PER_TILE * HEAD_DIM, min(n_keys - KEYS_PER_TILE, KEYS_PER_TILE));
#if UNICA_Q_SMEM
    __shared__ __align__(16) int8_t q_tiles[WARPS_PER_BLOCK][ROWS_PER_WARP * K8_STRIDE];
    load_q8_async(q_tiles[warp], Q8 + q_offset, Q8, n_rows, lane);
    cp_async_commit();
    auto qk = [&](float s[KEY_BLOCKS][4], const int8_t* k_tile) { qk_scores_int8_q_smem(s, q_tiles[warp], k_tile, lane); };
#else
    cp_async_commit();
    uint32_t q[Q8_STEPS][4];
    load_q8(q, Q8 + q_offset, row, col, n_rows);
    auto qk = [&](float s[KEY_BLOCKS][4], const int8_t* k_tile) { qk_scores_int8(s, q, k_tile, lane); };
#endif

    float row_max[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float row_sum[2] = {0.0f, 0.0f};
    float out[D_BLOCKS][4] = {};

    cp_async_wait();
    __syncthreads();
    float scores[2][KEY_BLOCKS][4];  // tile t in scores[t % 2]; Q·Kᵀ(t + 1) writes the other
    qk(scores[0], k_tiles[0]);
    if (n_keys < KEYS_PER_TILE) mask_scores(scores[0], col, n_keys);

    // Iteration t, with t % 2 == decltype(parity)::value known at compile time: every buffer and score index
    // is a constant, and the scores of t + 1 land where iteration t + 1 reads them, with no copy. A plain
    // loop with scores = next at its end spends 42 MOV per tile on that copy, and computes the buffer
    // addresses in every thread (M3.12).
    auto iteration = [&](int t, auto parity) {
        constexpr int now = decltype(parity)::value, later = 1 - now;
        const int key0 = t * KEYS_PER_TILE;
        cp_async_wait();  // this thread's part of K(t + 1) and V(t), issued an iteration ago
        __syncthreads();  // everyone's part; and every warp is done with K(t) and V(t − 1)
        if (t + 2 < n_tiles) {
            const int keys = key0 + 2 * KEYS_PER_TILE;
            load_tile8_async(k_tiles[now], K8 + keys * HEAD_DIM, min(n_keys - keys, KEYS_PER_TILE));
        }
        if (t + 1 < n_tiles) {
            const int keys = key0 + KEYS_PER_TILE;
            load_tile8_async<true>(v_tiles[later], V8 + keys * HEAD_DIM, min(n_keys - keys, KEYS_PER_TILE));
        }
        cp_async_commit();

        float alpha[2], beta[2];
        qk(scores[later], k_tiles[later]);  // stale K in the last iteration, never used
        online_softmax_u8(scores[now], row_max, row_sum, alpha, beta, q_scale_log2 * k_scale[key0 / K_SCALE_BLOCK]);
        uint32_t p[PV8_STEPS][4];
#pragma unroll
        for (int step = 0; step < PV8_STEPS; ++step) p_fragment_u8(p[step], scores[now], step);
        add_row_sum_u8(row_sum, p, alpha, beta);
        pv_int8(out, p, v_tiles[now], alpha, beta, lane);

        const int next_valid = n_keys - key0 - KEYS_PER_TILE;  // ≤ 0 in the last iteration: all masked
        if (next_valid < KEYS_PER_TILE) mask_scores(scores[later], col, next_valid);
    };

    int t = 0;
    for (; t + 1 < n_tiles; t += 2) {
        iteration(t, Parity<0>{});
        iteration(t + 1, Parity<1>{});
    }
    if (t < n_tiles) iteration(t, Parity<0>{});

    store_out(O + q_offset, out, row_sum, row, col, n_rows, v_scale);
}

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

// Waits until the phase with parity `phase` of the barrier has completed.
// M4.51: UNICA_WAIT_HINT > 0 (nanoseconds) adds the suspend-time hint: the retry becomes NANOSLEEP.SYNCS, the thread
// sleeps until a barrier event or the hint runs out, instead of issuing a check after every YIELD.
__device__ __forceinline__ void mbarrier_wait(uint64_t* barrier, int phase) {
    uint32_t done = 0;
    while (!done) {
        if constexpr (UNICA_WAIT_HINT > 0)
            asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3; selp.u32 %0, 1, 0, p; }"
                         : "=r"(done)
                         : "r"(smem_address(barrier)), "r"(phase), "n"(UNICA_WAIT_HINT)
                         : "memory");
        else
            asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                         : "=r"(done)
                         : "r"(smem_address(barrier)), "r"(phase)
                         : "memory");
    }
}

// One box of a 2D tensor map, starting at element x of row y, into shared memory (128-byte aligned); its bytes
// count towards `barrier`. Elements outside the tensor arrive as zeros. M4.48: the destination is .shared::cta. With
// .shared::cluster ptxas checks whether the address is another block's and, if so, calls a driver routine
// (__cuda_syscall_cp_async_bulk_*): never taken here, but an extern call in a kernel makes ptxas ignore setmaxnreg.
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap& map, int x, int y, uint64_t* barrier) {
    asm volatile(
        "cp.async.bulk.tensor.2d." COPY_DESTINATION ".global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        ::"r"(smem_address(dst)), "l"(&map), "r"(x), "r"(y), "r"(smem_address(barrier))
        : "memory");
}

// `bytes` (a multiple of 16) from global memory into shared memory (both 16-byte aligned), counted on `barrier`
__device__ __forceinline__ void bulk_load(void* dst, const void* src, int bytes, uint64_t* barrier) {
    asm volatile("cp.async.bulk." COPY_DESTINATION ".global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(smem_address(dst)), "l"(src), "r"(bytes), "r"(smem_address(barrier)) : "memory");
}

constexpr int K_TILE_BYTES = KEYS_PER_TILE * K8_STRIDE;  // a box of 144 bytes × KEYS_PER_TILE rows
constexpr bool TOKEN_TILES = UNICA_V_TOKEN && UNICA_TOKEN_COPY;
constexpr int TOK_TILE_BYTES = KEYS_PER_TILE / 2 * sizeof(float4);  // a tile's v_token, one float4 per pair of keys
constexpr int V_TILE_BYTES = UNICA_V_PAIRS ? KEYS_PER_TILE / 2 * V8_STRIDE : KEYS_PER_TILE * K8_STRIDE;

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
// k_map: K8 as rows of HEAD_DIM bytes, box K8_STRIDE × KEYS_PER_TILE. v_map: V8 in quantize_v layout 2
// (UNICA_V_PAIRS, uint16 rows of 2 · HEAD_DIM bytes, box V8_STRIDE bytes × KEYS_PER_TILE / 2) or layout 1 (as
// k_map). Q, scales, grid and block as attention_int8pv. The maps come first: TMA needs them 64-byte aligned,
// and a map after the 8-byte Q8 lands at offset 8 of the parameters (cudaErrorMisalignedAddress, M3.17).
// M4.48: with UNICA_PRODUCER the producer warpgroup's 128 threads come after the computing warps. 4 computing warps:
// the block launches at 128 registers per thread (2 blocks per SM), the computing warps take 232. M4.53, 8 computing
// warps (two warpgroups, 128 query rows sharing every K and V tile): 1 block per SM at 168 per thread, and the
// producer's (168 − 24) · 128 = 18,432 registers give the 256 computing threads 240 each.
constexpr int PRODUCER_THREADS = UNICA_PRODUCER ? 128 : 0;
constexpr int COMPUTING_REGISTERS = WARPS_PER_BLOCK == 8 ? 240 : 232;
static_assert(!UNICA_PRODUCER || WARPS_PER_BLOCK == 4 || WARPS_PER_BLOCK == 8,
              "setmaxnreg works on whole warpgroups: 4 or 8 computing warps");
extern "C" __global__ void __launch_bounds__(THREADS_PER_BLOCK + PRODUCER_THREADS,
                                             UNICA_PRODUCER ? 8 / WARPS_PER_BLOCK : UNICA_MIN_BLOCKS_PER_SM)
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
    const int row = lane / 4;
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
    constexpr int V_ROWS_PER_TILE = UNICA_V_PAIRS ? KEYS_PER_TILE / 2 : KEYS_PER_TILE;
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
    [[maybe_unused]] const int slot = TRACE_SLOT(lane);
    // M4.44 H5: next_max is the tile max of the tile in scores[t % 2], taken when its Q·Kᵀ was done. negligible[t % 2]
    // holds each warp's vote on tile t, written in iteration t − 1, read after iteration t's barrier; the slot is
    // written again for t + 2 only after the next barrier, when every warp has read it.
    [[maybe_unused]] float next_max[2];
    __shared__ __align__(4) uint8_t negligible[2][WARPS_PER_BLOCK];
    static_assert(!UNICA_SKIP_BLOCKS || WARPS_PER_BLOCK == 4, "the four votes are read as one word");
    if constexpr (UNICA_SKIP_BLOCKS > 0) tile_max_log2(scores[0], next_max, q_scale_log2 * k_scale[0]);

    // As in attention_int8pv_pipe. Iteration t reads K(t + 1) and V(t), issued in iteration t − 1 (or the
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

        TRACE_MARK(slot, lane, t, 0);
        bool skip_block = false;  // M4.44 H5: all four warps voted tile t negligible
        if constexpr (UNICA_SKIP_BLOCKS > 0)
            skip_block = t > 0 && *reinterpret_cast<const uint32_t*>(negligible[now]) == 0x01010101u;
        if (skip_block) {
            qk_scores_int8_q_smem(scores[later], q_tiles[warp], k_tiles[later], lane);
        } else {
            float alpha[2], beta[2];
            bool skipped = false;  // M4.44: stays false unless UNICA_SKIP_TILES
            [[maybe_unused]] float4 tok[KEY_BLOCKS];
            if constexpr (UNICA_V_TOKEN && !TOKEN_TILES) {  // from L2: every block of the head reads the same
                const float4* pairs = v_token + (size_t(blockIdx.y) * v8_tokens(n_keys) + key0 + col) / 2;
#pragma unroll
                for (int block = 0; block < KEY_BLOCKS; ++block) tok[block] = __ldg(pairs + block * MMA_N / 2);
            }
            qk_scores_int8_q_smem(scores[later], q_tiles[warp], k_tiles[later], lane);  // stale in the last iteration
            if constexpr (TOKEN_TILES)  // M4.41: landed with V(t), in this tile's buffer
                online_softmax_u8_token<MMA_N / 2>(scores[now], row_max, row_sum, alpha, beta,
                                                   q_scale_log2 * k_scale[key0 / K_SCALE_BLOCK], &tok_tiles[now][lane % 4]);
            else if constexpr (UNICA_V_TOKEN)
                online_softmax_u8_token<1>(scores[now], row_max, row_sum, alpha, beta,
                                           q_scale_log2 * k_scale[key0 / K_SCALE_BLOCK], tok);
            else if constexpr (UNICA_SKIP_BLOCKS > 0)
                online_softmax_u8_premax(scores[now], next_max, row_max, alpha, beta, q_scale_log2 * k_scale[key0 / K_SCALE_BLOCK]);
            else
                online_softmax_u8(scores[now], row_max, row_sum, alpha, beta, q_scale_log2 * k_scale[key0 / K_SCALE_BLOCK]);
            // M4.44: beta = 2^(tile max − running max) < 2^−n in both rows means the tile did not move the max (alpha is 1)
            // and weighs < 2^−n: the warp skips its P·V and the flush, and beta = 0 makes add_row_sum_u8 add nothing. The
            // only branch is the one before P·V, where pv_int8's rescale vote has one already: a branch earlier splits the
            // softmax from Q·Kᵀ(t + 1) into separate basic blocks, and ptxas no longer interleaves the two (+3…5 % with
            // nothing skipped, measured).
            if constexpr (UNICA_SKIP_TILES > 0) {
                skipped = __all_sync(0xffffffff, beta[0] < 0x1p0f / (1 << UNICA_SKIP_TILES) &&
                                                     beta[1] < 0x1p0f / (1 << UNICA_SKIP_TILES));
                beta[0] = skipped ? 0.0f : beta[0];
                beta[1] = skipped ? 0.0f : beta[1];
            }
            TRACE_MARK_AFTER(slot, lane, t, 1, __float_as_uint(alpha[0]) ^ __float_as_uint(alpha[1]));
            uint32_t p[PV8_STEPS][4];
#pragma unroll
            for (int step = 0; step < PV8_STEPS; ++step) p_fragment_u8(p[step], scores[now], step);
            TRACE_MARK_AFTER(slot, lane, t, 3, p[0][0] ^ p[0][1] ^ p[0][2] ^ p[0][3] ^ p[1][0] ^ p[1][1] ^ p[1][2] ^ p[1][3]);
            if constexpr (!UNICA_V_TOKEN) add_row_sum_u8(row_sum, p, alpha, beta);
            if (!skipped) pv_int8<UNICA_V_PAIRS>(out, p, v_tiles[now], alpha, beta, lane);
            // M4.45 A3: point 2 here is P·V done (out flushed), where the warp heads for the barrier
            TRACE_MARK_AFTER(slot, lane, t, 2, __float_as_uint(out[0][0]) ^ __float_as_uint(out[D_BLOCKS - 1][3]));
        }

        const int next_valid = n_keys - key0 - KEYS_PER_TILE;  // ≤ 0 in the last iteration: all masked
        if (next_valid < KEYS_PER_TILE)
            mask_scores(scores[later], UNICA_RECOMPUTE ? 2 * (fresh_thread_index() % 4) : col, next_valid);
        if constexpr (UNICA_SKIP_BLOCKS > 0) {  // the vote on tile t + 1, against the running max after tile t
            if (t + 1 < n_tiles) {
                tile_max_log2(scores[later], next_max, q_scale_log2 * k_scale[(key0 + KEYS_PER_TILE) / K_SCALE_BLOCK]);
                const bool low = next_max[0] < row_max[0] - UNICA_SKIP_BLOCKS && next_max[1] < row_max[1] - UNICA_SKIP_BLOCKS;
                const bool all = __all_sync(0xffffffff, low);
                negligible[later][warp] = all;  // every lane the same byte: no branch, no reconvergence
            }
        }
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
    if constexpr (UNICA_RECOMPUTE) {
        const int index = fresh_thread_index(), at = index % WARP_SIZE;
        const int first = blockIdx.x * ROWS_PER_BLOCK + index / WARP_SIZE * ROWS_PER_WARP;  // row0
        store_out(O + o_layout.offset(blockIdx.y, first), out, row_sum, at / 4, 2 * (at % 4),
                  min(n_queries - first, ROWS_PER_WARP), v_scale, o_layout.token);
    } else {
        store_out(O + o_layout.offset(blockIdx.y, row0), out, row_sum, row, col, n_rows, v_scale, o_layout.token);
    }
}

// ---------------------------------------------------------------------------------------------------
// M4.8: P·V in NVFP4 (e2m1 values, an e4m3 scale per 16 keys), mma.sync m16n8k64 kind::mxf4nvf4, which runs
// at twice the 8-bit rate and accumulates in f32 itself (LOG.md M4.1).
// ---------------------------------------------------------------------------------------------------

constexpr int MMA_K4 = 64;          // keys in one m16n8k64: a whole tile
constexpr int FP4_SCALE_BLOCK = 16;  // keys per e4m3 scale
// attention_fp4pv takes a tile of 64 keys in one mma, so it exists only in builds with KEYS_PER_TILE = 64
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

#if UNICA_KEYS_PER_TILE == 64
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

// M4.14 (H9c): V4 tiles (and attention_fp4's K4 tiles) copied as one contiguous block each (cp.async.bulk, 1D) instead of a 2D TMA box of
// 64 rows of 80 bytes and 128 rows of 48: both tiles lie in one piece in global memory, K4 padded to whole tiles
#ifndef UNICA_BULK_COPY
#define UNICA_BULK_COPY 1
#endif

// M4.8: attention_int8pv_tma with P·V in NVFP4. Q·Kᵀ, its pipelining and the K tiles are unchanged. v_map: V4
// (quantize_v4) as rows of V4_ROW_BYTES, box V4_ROW_BYTES × HEAD_DIM: one tile. v_outer: the float scale per
// channel, applied in store_out. Arguments, grid and block as attention_int8pv_tma.
extern "C" __global__ void __launch_bounds__(THREADS_PER_BLOCK, UNICA_MIN_BLOCKS_PER_SM)
    attention_fp4pv(const __grid_constant__ CUtensorMap k_map, const __grid_constant__ CUtensorMap v_map,
                    const uint8_t* V4, const int8_t* Q8, const float* q_scale, const float* k_scale, const float* v_outer,
                    __nv_bfloat16* O, int n_queries, int n_keys, float scale_log2, TokenLayout o_layout) {
    __shared__ __align__(128) int8_t k_tiles[2][K_TILE_BYTES];
    __shared__ __align__(128) uint8_t v_tiles[2][V4_TILE_BYTES];
    __shared__ __align__(16) int8_t q_tiles[WARPS_PER_BLOCK][ROWS_PER_WARP * K8_STRIDE];
    __shared__ __align__(8) uint64_t landed[3];

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int row = lane / 4;
    const int col = 2 * (lane % 4);

    const int row0 = blockIdx.x * ROWS_PER_BLOCK + warp * ROWS_PER_WARP;
    const int n_rows = min(n_queries - row0, ROWS_PER_WARP);

    const size_t q_offset = (size_t(blockIdx.y) * n_queries + row0) * HEAD_DIM;
    v_outer += size_t(blockIdx.y) * HEAD_DIM;
    k_scale += size_t(blockIdx.y) * ((n_keys + K_SCALE_BLOCK - 1) / K_SCALE_BLOCK);
    const float q_scale_log2 =
        q_scale[size_t(blockIdx.y) * ((n_queries + Q_SCALE_BLOCK - 1) / Q_SCALE_BLOCK) + row0 / Q_SCALE_BLOCK] *
        scale_log2;
    const int n_tiles = (n_keys + KEYS_PER_TILE - 1) / KEYS_PER_TILE;

    const int k_row0 = blockIdx.y * n_keys;
    const int v_row0 = blockIdx.y * n_tiles * HEAD_DIM;
    auto load_k = [&](int buffer, int tile, uint64_t* barrier) {
        tma_load_2d(k_tiles[buffer], k_map, 0, k_row0 + tile * KEYS_PER_TILE, barrier);
    };
    auto load_v = [&](int buffer, int tile, uint64_t* barrier) {
        if (UNICA_BULK_COPY)
            bulk_load(v_tiles[buffer], V4 + (size_t(blockIdx.y) * n_tiles + tile) * V4_TILE_BYTES, V4_TILE_BYTES, barrier);
        else
            tma_load_2d(v_tiles[buffer], v_map, 0, v_row0 + tile * HEAD_DIM, barrier);
    };

    if (threadIdx.x == 0) {
        for (int i = 0; i < 3; ++i) mbarrier_init(&landed[i], 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        mbarrier_expect_bytes(&landed[2], (n_tiles > 1 ? 2 : 1) * K_TILE_BYTES + V4_TILE_BYTES);
        load_k(0, 0, &landed[2]);
        load_v(0, 0, &landed[2]);
        if (n_tiles > 1) load_k(1, 1, &landed[2]);
    }
    load_q8_async(q_tiles[warp], Q8 + q_offset, Q8, n_rows, lane);
    cp_async_commit();

    float row_max[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float row_sum_acc[4] = {};
    float out[D_BLOCKS][4] = {};

    cp_async_wait();
    mbarrier_wait(&landed[2], 0);
    __syncthreads();
    float scores[2][KEY_BLOCKS][4];
    qk_scores_int8_q_smem(scores[0], q_tiles[warp], k_tiles[0], lane);
    if (n_keys < KEYS_PER_TILE) mask_scores(scores[0], col, n_keys);
    [[maybe_unused]] const int slot = TRACE_SLOT(lane);

    auto iteration = [&](int t, auto parity) {
        constexpr int now = decltype(parity)::value, later = 1 - now;
        const int key0 = t * KEYS_PER_TILE;
        if (t > 0) mbarrier_wait(&landed[later], (t - 1) / 2 % 2);
        __syncthreads();
        if (threadIdx.x == 0 && t + 1 < n_tiles) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_expect_bytes(&landed[now], (t + 2 < n_tiles ? K_TILE_BYTES : 0) + V4_TILE_BYTES);
            if (t + 2 < n_tiles) load_k(now, t + 2, &landed[now]);
            load_v(later, t + 1, &landed[now]);
        }

        float alpha[2];
        uint32_t p[4], scale_p;
        TRACE_MARK(slot, lane, t, 0);
        qk_scores_int8_q_smem(scores[later], q_tiles[warp], k_tiles[later], lane);  // stale in the last iteration
        online_softmax_fp4(scores[now], row_max, alpha, p, scale_p, q_scale_log2 * k_scale[key0 / K_SCALE_BLOCK], lane);
        TRACE_MARK_AFTER(slot, lane, t, 1, __float_as_uint(alpha[0]) ^ __float_as_uint(alpha[1]));
        TRACE_MARK_AFTER(slot, lane, t, 2, scale_p);
        TRACE_MARK_AFTER(slot, lane, t, 3, p[0] ^ p[1] ^ p[2] ^ p[3]);
        pv_fp4(out, row_sum_acc, p, scale_p, v_tiles[now], alpha, lane);

        const int next_valid = n_keys - key0 - KEYS_PER_TILE;
        if (next_valid < KEYS_PER_TILE) mask_scores(scores[later], col, next_valid);
    };

    int t = 0;
    for (; t + 1 < n_tiles; t += 2) {
        iteration(t, Parity<0>{});
        iteration(t + 1, Parity<1>{});
    }
    if (t < n_tiles) iteration(t, Parity<0>{});

    const float row_sum[2] = {row_sum_acc[0], row_sum_acc[2]};  // 0 outside lane 4g: store_out's quad sum adds them
    store_out(O + o_layout.offset(blockIdx.y, row0), out, row_sum, row, col, n_rows, v_outer, o_layout.token);
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
#ifndef UNICA_STAGES
#define UNICA_STAGES 2
#endif
constexpr int STAGES = UNICA_STAGES;
constexpr int AHEAD = STAGES - 1;

// M4.21: tiles of smooth_bias per copy. 1: a tile's 256 bytes travel in K's group, one more bulk copy per tile, and
// that copy, not its bytes, cost ~2 ms of the +4.5 (neither an L2 prefetch nor a deeper pipeline helped, nor issuing it
// from another warp). n: one copy every n tiles into one of two buffers (2n · 256 bytes), issued n tiles ahead;
// 8: +3.2 ms, 32: +2.3 ms, the rest is the 16 LDS a tile (ncu: +24 instructions a tile, no MIO throttle).
#ifndef UNICA_BIAS_CHUNK
#define UNICA_BIAS_CHUNK 32
#endif
constexpr int BIAS_CHUNK_TILES = UNICA_BIAS_CHUNK;
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
extern "C" __global__ void __launch_bounds__(THREADS_PER_BLOCK, UNICA_MIN_BLOCKS_PER_SM)
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

    // K4 is padded to whole tiles per head (its padding rows are zeros), like V4
    const int k_row0 = blockIdx.y * n_tiles * KEYS_PER_TILE;
    const int v_row0 = blockIdx.y * n_tiles * HEAD_DIM;
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
    auto load_k = [&](int buffer, int tile, uint64_t* barrier) {
        if (UNICA_BULK_COPY)
            bulk_load(k_tiles[buffer], K4 + (size_t(blockIdx.y) * n_tiles + tile) * K4_TILE_BYTES, K4_TILE_BYTES, barrier);
        else
            tma_load_2d(k_tiles[buffer], k_map, 0, k_row0 + tile * KEYS_PER_TILE, barrier);
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
        if (UNICA_BULK_COPY)
            bulk_load(v_tiles[buffer], V4 + (size_t(blockIdx.y) * n_tiles + tile) * V4_TILE_BYTES, V4_TILE_BYTES, barrier);
        else
            tma_load_2d(v_tiles[buffer], v_map, 0, v_row0 + tile * HEAD_DIM, barrier);
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
    [[maybe_unused]] const int slot = TRACE_SLOT(lane);

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
        TRACE_MARK(slot, lane, t, 0);
        qk_scores_fp4(scores[later], q, q_scale, k_tiles[(t + 1) % STAGES], bias_tile(t + 1),
                      lane);  // stale in the last iteration
        online_softmax_fp4(scores[now], row_max, alpha, p, scale_p, score_scale, lane);
        TRACE_MARK_AFTER(slot, lane, t, 1, __float_as_uint(alpha[0]) ^ __float_as_uint(alpha[1]));
        TRACE_MARK_AFTER(slot, lane, t, 2, scale_p);
        TRACE_MARK_AFTER(slot, lane, t, 3, p[0] ^ p[1] ^ p[2] ^ p[3]);
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
#endif  // UNICA_KEYS_PER_TILE == 64
