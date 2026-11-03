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

// The same mma with fp16 inputs, same layout and same rate. Used for P·V: P lies in [0, 1] and needs
// precision, not range, and fp16 has 10 mantissa bits to bf16's 7.
__device__ __forceinline__ void mma_f16_16816(float acc[4], const uint32_t a[4], const uint32_t b[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
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
__device__ __forceinline__ void store_out(__nv_bfloat16* O, const float out[D_BLOCKS][4], const float row_sum[2],
                                          int row, int col, int n_rows) {
    const float inv_sum[2] = {1.0f / quad_sum(row_sum[0]), 1.0f / quad_sum(row_sum[1])};
#pragma unroll
    for (int block = 0; block < D_BLOCKS; ++block) {
        const int dim = block * MMA_N + col;
        if (row < n_rows)
            *reinterpret_cast<__nv_bfloat162*>(O + row * HEAD_DIM + dim) =
                __floats2bfloat162_rn(out[block][0] * inv_sum[0], out[block][1] * inv_sum[0]);
        if (row + 8 < n_rows)
            *reinterpret_cast<__nv_bfloat162*>(O + (row + 8) * HEAD_DIM + dim) =
                __floats2bfloat162_rn(out[block][2] * inv_sum[1], out[block][3] * inv_sum[1]);
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

constexpr int WARPS_PER_BLOCK = 4;
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
__device__ __forceinline__ void ldmatrix_x4(uint32_t r[4], const __nv_bfloat16* lane_row) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_address(lane_row)));
}

// The same, transposed on the way in: the lane gets rows 2 · (lane % 4) and + 1 of column lane / 4.
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t r[4], const __nv_bfloat16* lane_row) {
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
// Same grid, block and arguments as attention.
extern "C" __global__ void attention_smem(const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
                                          __nv_bfloat16* O, int n_queries, int n_keys, float scale_log2) {
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
