/*
 * M1: plain flash attention on mma.sync. Q·Kᵀ in BF16, P·V in FP16, both accumulate in FP32.
 *
 * Built up one piece at a time: one mma, a Q·Kᵀ tile, online softmax, the P·V loop, the whole
 * attention. Each piece keeps its own test kernel, so a broken layout shows up where it was made.
 */

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
                                          int row, int col, int n_rows, const float* v_scale = nullptr) {
    const float inv_sum[2] = {1.0f / quad_sum(row_sum[0]), 1.0f / quad_sum(row_sum[1])};
#pragma unroll
    for (int block = 0; block < D_BLOCKS; ++block) {
        const int dim = block * MMA_N + col;
        const float scale[2] = {v_scale ? v_scale[dim] : 1.0f, v_scale ? v_scale[dim + 1] : 1.0f};
        if (row < n_rows)
            *reinterpret_cast<__nv_bfloat162*>(O + row * HEAD_DIM + dim) = __floats2bfloat162_rn(
                out[block][0] * inv_sum[0] * scale[0], out[block][1] * inv_sum[0] * scale[1]);
        if (row + 8 < n_rows)
            *reinterpret_cast<__nv_bfloat162*>(O + (row + 8) * HEAD_DIM + dim) = __floats2bfloat162_rn(
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

// Quantization for attention_int8qk. One block of threads per scale block: block_tokens tokens of one head,
// x ≈ x8 · scale with x = X − mean (mean [heads][HEAD_DIM], or null). Bit for bit what bench/emulate.py's
// quant() computes: the same max, the same IEEE division (nvcc's default), round half to even like
// torch.round. The block is read twice; the second read comes from L2, not from VRAM.
// Grid: (ceil(n_tokens / block_tokens), heads). Block: QUANTIZE_THREADS.
extern "C" __global__ void __launch_bounds__(QUANTIZE_THREADS)
    quantize_int8(const __nv_bfloat16* X, const float* mean, int8_t* X8, float* scale, int n_tokens, int block_tokens) {
    const int token0 = blockIdx.x * block_tokens;
    const int n_chunks = min(block_tokens, n_tokens - token0) * CHUNKS_PER_ROW;
    const size_t offset = (size_t(blockIdx.y) * n_tokens + token0) * HEAD_DIM;
    X += offset;
    X8 += offset;
    if (mean) mean += blockIdx.y * HEAD_DIM;

    float local_max = 0.0f;
    for (int chunk = threadIdx.x; chunk < n_chunks; chunk += QUANTIZE_THREADS) {
        const int dim = chunk % CHUNKS_PER_ROW * CHUNK;
        float values[CHUNK];
        load_chunk(values, X + chunk * CHUNK, mean ? mean + dim : nullptr);
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
        load_chunk(values, X + chunk * CHUNK, mean ? mean + dim : nullptr);
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
// carries the same 255, and the two cancel in out / sum.
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
        float tile_sum = 0.0f;
#pragma unroll
        for (int block = 0; block < KEY_BLOCKS; ++block)
#pragma unroll
            for (int i = 2 * half; i < 2 * half + 2; ++i) {
                scores[block][i] = exp2_ftz(fmaf(scores[block][i], scale_log2, offset));
                tile_sum += scores[block][i];
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
// so 8 int32 accumulators are alive at a time, not 64.
__device__ __forceinline__ void pv_int8(float out[D_BLOCKS][4], const uint32_t p[PV8_STEPS][4], const int8_t* v_tile,
                                        const float alpha[2], const float beta[2], int lane) {
#if UNICA_LAZY_RESCALE
    // M3.9: alpha is exactly 1 unless the max of a row grew in this tile, and then out · alpha is out. When it
    // is 1 in every row of the warp, the 64 multiplies are skipped. Rounding is the same on both paths.
    if (!__all_sync(0xffffffff, alpha[0] == 1.0f && alpha[1] == 1.0f)) rescale_out(out, alpha);
#endif
    const int8_t* lane_row = v_tile + lane * K8_STRIDE;  // keys 0..15 of a step for matrix 0, 16..31 for matrix 1
#pragma unroll
    for (int pair = 0; pair < D_BLOCKS / 2; ++pair) {
        int acc[2][4] = {};
#pragma unroll
        for (int step = 0; step < PV8_STEPS; ++step) {
            uint32_t b[4];
            ldmatrix_x2_trans_b8(b, lane_row + step * MMA_K8 * K8_STRIDE + pair * 2 * MMA_N);
            const uint32_t left[2] = {b[0], b[2]}, right[2] = {b[1], b[3]};
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

// V for attention_int8pv: v ≈ v8 · v_scale[dim], one scale per channel over all tokens of a head. d is the
// axis P·V does not sum over, so the scale stays out of the mma and store_out applies it once. v_scale comes
// from torch: the max of a channel needs every token. Bit for bit bench/emulate.py: IEEE division, round half
// to even, clamp to ±127. Grid: ceil(n_chunks / QUANTIZE_THREADS), n_chunks = heads · n_tokens · CHUNKS_PER_ROW.
extern "C" __global__ void __launch_bounds__(QUANTIZE_THREADS)
    quantize_v(const __nv_bfloat16* V, const float* v_scale, int8_t* V8, int n_tokens, int n_chunks) {
    const int chunk = blockIdx.x * QUANTIZE_THREADS + threadIdx.x;
    if (chunk >= n_chunks) return;
    const float* scale = v_scale + chunk / (n_tokens * CHUNKS_PER_ROW) * HEAD_DIM + chunk % CHUNKS_PER_ROW * CHUNK;
    float values[CHUNK];
    load_chunk(values, V + size_t(chunk) * CHUNK, nullptr);
    uint32_t packed[2] = {0, 0};
#pragma unroll
    for (int i = 0; i < CHUNK; ++i) {
        const int value = __float2int_rn(fminf(fmaxf(rintf(values[i] / scale[i]), -127.0f), 127.0f));
        packed[i / 4] |= uint32_t(value & 0xff) << (8 * (i % 4));
    }
    *reinterpret_cast<uint2*>(V8 + size_t(chunk) * CHUNK) = make_uint2(packed[0], packed[1]);
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
#else
#define TRACE_SLOT(lane) (-1)
#define TRACE_MARK(slot, lane, tile, point) ((void)0)
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
