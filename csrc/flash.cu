/*
 * M1: plain flash attention on mma.sync, BF16 inputs, FP32 accumulate.
 *
 * Built up one piece at a time: one mma, a Q·Kᵀ tile, online softmax, the P·V loop.
 * Each piece keeps its own test kernel, so a broken layout shows up where it was made.
 */

#include <cuda_bf16.h>
#include <math_constants.h>
#include <cstdint>

// Shape of one mma.sync.m16n8k16: D[16×8] += A[16×16] · B[16×8].
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

constexpr int HEAD_DIM = 128;      // WAN 2.2
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

constexpr int Q_STEPS = HEAD_DIM / MMA_K;          // 8 fragments of Q along d
constexpr int KEY_BLOCKS = KEYS_PER_TILE / MMA_N;  // 8 blocks of 16 × 8 in one score tile

// 16 rows of Q as A fragments, kept in registers for the whole pass over K: 32 registers per lane.
__device__ __forceinline__ void load_q(uint32_t q[Q_STEPS][4], const __nv_bfloat16* Q, int row, int col) {
#pragma unroll
    for (int step = 0; step < Q_STEPS; ++step) load_a(q[step], Q + step * MMA_K, HEAD_DIM, row, col);
}

// scores = Q · Kᵀ for one tile of KEYS_PER_TILE keys: 8 blocks of 16 × 8, 32 floats per lane.
// The loops must unroll, otherwise scores[block] with a run-time block cannot live in registers
// and goes to local memory.
__device__ __forceinline__ void qk_scores(float scores[KEY_BLOCKS][4], const uint32_t q[Q_STEPS][4],
                                          const __nv_bfloat16* K, int row, int col) {
#pragma unroll
    for (int block = 0; block < KEY_BLOCKS; ++block)
#pragma unroll
        for (int i = 0; i < 4; ++i) scores[block][i] = 0.0f;
#pragma unroll
    for (int step = 0; step < Q_STEPS; ++step)
#pragma unroll
        for (int block = 0; block < KEY_BLOCKS; ++block) {
            uint32_t b[2];
            load_b(b, K + block * MMA_N * HEAD_DIM + step * MMA_K, HEAD_DIM, row, col);
            mma_bf16_16816(scores[block], q[step], b);
        }
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
// n_keys must be a multiple of KEYS_PER_TILE: no masking yet. Launch with a single warp.
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

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    const __nv_bfloat162 pair = __floats2bfloat162_rn(lo, hi);  // lo goes to the low 16 bits, the lower address
    return *reinterpret_cast<const uint32_t*>(&pair);
}

__device__ __forceinline__ uint32_t pack_bf16(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    const __nv_bfloat162 pair = __halves2bfloat162(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&pair);
}

// B fragment of V for 16 keys × 8 columns of d: B[n][k] = V[k][n]. The two keys of one register sit
// HEAD_DIM apart in V, so it takes two 16-bit loads instead of one 32-bit load. ldmatrix.trans
// transposes on the way in and replaces this in M2.
__device__ __forceinline__ void load_v(uint32_t b[2], const __nv_bfloat16* V, int row, int col) {
    b[0] = pack_bf16(V[col * HEAD_DIM + row], V[(col + 1) * HEAD_DIM + row]);
    b[1] = pack_bf16(V[(col + 8) * HEAD_DIM + row], V[(col + 9) * HEAD_DIM + row]);
}

// Step 4: attention for 16 query rows, O = softmax(Q · Kᵀ / sqrt(HEAD_DIM)) · V, all keys.
// Q is 16 × HEAD_DIM; K and V are n_keys × HEAD_DIM; O is 16 × HEAD_DIM fp32.
// n_keys must be a multiple of KEYS_PER_TILE: no masking yet. Launch with a single warp.
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

    for (int key0 = 0; key0 < n_keys; key0 += KEYS_PER_TILE) {
        float scores[KEY_BLOCKS][4], alpha[2];
        qk_scores(scores, q, K + key0 * HEAD_DIM, row, col);
        online_softmax(scores, row_max, row_sum, alpha, scale_log2);

        // what O has gathered so far was weighted against the old max
#pragma unroll
        for (int block = 0; block < D_BLOCKS; ++block) {
            out[block][0] *= alpha[0];
            out[block][1] *= alpha[0];
            out[block][2] *= alpha[1];
            out[block][3] *= alpha[1];
        }

        // Two neighbouring 16 × 8 blocks of P, as Q·Kᵀ left them, are exactly one 16 × 16 A fragment:
        // P goes from output to input without leaving the lane.
#pragma unroll
        for (int step = 0; step < KEY_STEPS; ++step) {
            const float* left = scores[2 * step];
            const float* right = scores[2 * step + 1];
            const uint32_t p[4] = {
                pack_bf16(left[0], left[1]),
                pack_bf16(left[2], left[3]),
                pack_bf16(right[0], right[1]),
                pack_bf16(right[2], right[3]),
            };
            const __nv_bfloat16* v_rows = V + (key0 + step * MMA_K) * HEAD_DIM;
#pragma unroll
            for (int block = 0; block < D_BLOCKS; ++block) {
                uint32_t b[2];
                load_v(b, v_rows + block * MMA_N, row, col);
                mma_bf16_16816(out[block], p, b);
            }
        }
    }

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
