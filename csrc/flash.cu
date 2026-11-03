/*
 * M1: plain flash attention on mma.sync, BF16 inputs, FP32 accumulate.
 *
 * Built up one piece at a time: one mma, a Q·Kᵀ tile, online softmax, the P·V loop.
 * Each piece keeps its own test kernel, so a broken layout shows up where it was made.
 */

#include <cuda_bf16.h>
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

// Step 2: S = Q · Kᵀ for 16 query rows and KEYS_PER_TILE keys, summed over HEAD_DIM.
// Q is 16 × HEAD_DIM, K is KEYS_PER_TILE × HEAD_DIM, S is 16 × KEYS_PER_TILE fp32. Launch with a single warp.
extern "C" __global__ void qk_tile(const __nv_bfloat16* Q, const __nv_bfloat16* K, float* S) {
    const int lane = threadIdx.x;
    const int row = lane / 4;
    const int col = 2 * (lane % 4);

    // 8 blocks of 16 × 8 side by side: 32 floats per lane. The loops below must unroll,
    // otherwise acc[n] with a run-time n cannot live in registers and goes to local memory.
    float acc[KEYS_PER_TILE / MMA_N][4] = {};

#pragma unroll
    for (int k0 = 0; k0 < HEAD_DIM; k0 += MMA_K) {
        uint32_t a[4];
        load_a(a, Q + k0, HEAD_DIM, row, col);  // loaded once, used by all 8 key blocks
#pragma unroll
        for (int n = 0; n < KEYS_PER_TILE / MMA_N; ++n) {
            uint32_t b[2];
            load_b(b, K + n * MMA_N * HEAD_DIM + k0, HEAD_DIM, row, col);
            mma_bf16_16816(acc[n], a, b);
        }
    }

#pragma unroll
    for (int n = 0; n < KEYS_PER_TILE / MMA_N; ++n) store_acc(S + n * MMA_N, KEYS_PER_TILE, acc[n], row, col);
}
