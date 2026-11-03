/*
 * Peak rates of the instructions attention is built from, measured on this card.
 *
 * Each kernel runs one instruction form in 8 independent chains per warp, so no instruction waits for
 * the one before it. Inputs are 1.0 (or 1) in every format: values do not change the rate, they only keep
 * the accumulators finite. out keeps the results alive so nothing is optimised away.
 */

#include <cstdint>

constexpr int CHAINS = 8;

// mma with a 4-register accumulator: f32 or s32 (the constraint picks which).
#define MMA4(PTX, C, acc, a, b)                                                              \
    asm(PTX " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"                        \
        : "+" C(acc[0]), "+" C(acc[1]), "+" C(acc[2]), "+" C(acc[3])                       \
        : "r"(a), "r"(a), "r"(a), "r"(a), "r"(b), "r"(b))

// mma with an f16 accumulator: two registers of f16x2.
#define MMA2(PTX, acc, a, b)                                                                 \
    asm(PTX " {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"                                    \
        : "+r"(acc[0]), "+r"(acc[1])                                                        \
        : "r"(a), "r"(a), "r"(a), "r"(a), "r"(b), "r"(b))

#define PEAK_KERNEL(NAME, ACC_T, ACC_REGS, BODY)                                             \
    extern "C" __global__ void NAME(float* out, int iterations) {                           \
        ACC_T acc[CHAINS][ACC_REGS] = {};                                                    \
        for (int i = 0; i < iterations; ++i) {                                               \
            _Pragma("unroll") for (int chain = 0; chain < CHAINS; ++chain) { BODY; }        \
        }                                                                                    \
        float sum = 0.0f;                                                                    \
        _Pragma("unroll") for (int chain = 0; chain < CHAINS; ++chain)                      \
            _Pragma("unroll") for (int r = 0; r < ACC_REGS; ++r) sum += float(acc[chain][r]); \
        out[blockIdx.x * blockDim.x + threadIdx.x] = sum;                                    \
    }

constexpr uint32_t ONES_BF16 = 0x3f803f80, ONES_F16 = 0x3c003c00;
constexpr uint32_t ONES_INT8 = 0x01010101, ONES_E4M3 = 0x38383838;

// m16n8k16, 4096 flop each: the exact path (half bf16 Q·Kᵀ, half fp16 P·V), FP32 accumulate.
PEAK_KERNEL(peak_bf16_f16_f32, float, 4,
            if (chain % 2) MMA4("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", "f", acc[chain], ONES_F16, ONES_F16);
            else MMA4("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", "f", acc[chain], ONES_BF16, ONES_BF16))

// m16n8k32, 8192 flop each.
PEAK_KERNEL(peak_s8s8, int, 4,
            MMA4("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32", "r", acc[chain], ONES_INT8, ONES_INT8))
PEAK_KERNEL(peak_u8s8, int, 4,
            MMA4("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32", "r", acc[chain], ONES_INT8, ONES_INT8))
PEAK_KERNEL(peak_e4m3_f32, float, 4,
            MMA4("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32", "f", acc[chain], ONES_E4M3, ONES_E4M3))
PEAK_KERNEL(peak_e4m3_f16, uint32_t, 2,
            MMA2("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16", acc[chain], ONES_E4M3, ONES_E4M3))

// The two candidate kernels' mixes: Q·Kᵀ in s8 × s8 alternating with P·V in u8 × s8, or in e4m3 with
// an f16 accumulator.
PEAK_KERNEL(peak_mix_s8s8_u8s8, int, 4,
            if (chain % 2) MMA4("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32", "r", acc[chain], ONES_INT8, ONES_INT8);
            else MMA4("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32", "r", acc[chain], ONES_INT8, ONES_INT8))
PEAK_KERNEL(peak_mix_s8s8_e4m3_f16, int, 4,
            if (chain % 2) MMA2("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16", acc[chain], ONES_E4M3, ONES_E4M3);
            else MMA4("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32", "r", acc[chain], ONES_INT8, ONES_INT8))

// Conversions the two P·V variants need, per 32-bit result: int32 -> f32 (flushing the u8 × s8
// accumulator), f16x2 -> 2 × f32 (flushing the f16 one), and the exponent for scale.
// threadIdx.x keeps the value per lane: a warp-uniform one goes to the uniform datapath instead
PEAK_KERNEL(peak_i2f, float, 1, acc[chain][0] += __int2float_rn(i * CHAINS + chain - int(threadIdx.x)))
PEAK_KERNEL(peak_ex2, float, 1, asm("ex2.approx.ftz.f32 %0, %0;" : "+f"(acc[chain][0])))

// M3.8: the u8 × s8 mma the way attention issues it. Four different A registers and a B pair of its own per
// chain, as fragments from ldmatrix are, and FILLERS FFMA per mma in a chain of their own, standing in for
// the rest of a tile's work. salt is 0 at run time (no device address has its top bit set), but the
// compiler cannot know it, so every operand stays a register of its own and the values stay 1. It differs
// per lane as far as the compiler knows, as ldmatrix fragments do: a warp-uniform salt puts B in uniform
// registers and adds a MOV per mma. (A salt from iterations does not work: inside the loop iterations > 0,
// and ptxas concludes iterations >> 31 == 0.)
enum Filler { FFMA, FFMA_IMM, FMUL, FMNMX, IADD, I2F, EX2 };

// One filler instruction of the given kind on x, in a dependent chain; y is a run-time operand, different for
// each step. Written in PTX, and x starts from a run-time value, because the compiler otherwise folds
// chains it can see through (x · 0.5 eight times becomes one multiply, a chain from 0 becomes a constant).
template <Filler KIND>
__device__ __forceinline__ void filler_step(float& x, float y) {
    if constexpr (KIND == FFMA) asm volatile("fma.rn.f32 %0, %0, %1, %1;" : "+f"(x) : "f"(y));
    if constexpr (KIND == FFMA_IMM) asm volatile("fma.rn.f32 %0, %0, 0f3F000000, 0f3F800000;" : "+f"(x));  // x · 0.5 + 1
    if constexpr (KIND == FMUL) asm volatile("mul.rn.f32 %0, %0, %1;" : "+f"(x) : "f"(y));
    if constexpr (KIND == FMNMX) asm volatile("max.f32 %0, %0, %1;" : "+f"(x) : "f"(y));
    if constexpr (KIND == IADD) {
        int bits = __float_as_int(x);
        asm volatile("add.s32 %0, %0, %1;" : "+r"(bits) : "r"(__float_as_int(y)));
        x = __int_as_float(bits);
    }
    if constexpr (KIND == I2F) asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(x) : "r"(__float_as_int(x)));
    if constexpr (KIND == EX2) asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(x));
}

template <int FILLERS, Filler KIND = FFMA>
__device__ __forceinline__ void mixed_u8s8(float* out, int iterations) {
    const uint32_t salt = uint32_t(reinterpret_cast<uintptr_t>(out) >> 63) * threadIdx.x;
    uint32_t a[4], b[CHAINS][2];
#pragma unroll
    for (int r = 0; r < 4; ++r) a[r] = ONES_INT8 | salt << r;
#pragma unroll
    for (int chain = 0; chain < CHAINS; ++chain) {
        b[chain][0] = ONES_INT8 | salt << (4 + chain);
        b[chain][1] = ONES_INT8 | salt << (12 + chain);
    }
    int acc[CHAINS][4] = {};
    float filler[CHAINS], operand[FILLERS > 0 ? FILLERS : 1];
#pragma unroll
    for (int chain = 0; chain < CHAINS; ++chain) filler[chain] = __uint_as_float(salt << chain);
#pragma unroll
    for (int f = 0; f < (FILLERS > 0 ? FILLERS : 1); ++f) operand[f] = __uint_as_float(salt << (f + 8));
    for (int i = 0; i < iterations; ++i) {
#pragma unroll
        for (int chain = 0; chain < CHAINS; ++chain) {
            asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+r"(acc[chain][0]), "+r"(acc[chain][1]), "+r"(acc[chain][2]), "+r"(acc[chain][3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[chain][0]), "r"(b[chain][1]));
#pragma unroll
            for (int f = 0; f < FILLERS; ++f) filler_step<KIND>(filler[chain], operand[f]);
        }
    }
    float sum = 0.0f;
#pragma unroll
    for (int chain = 0; chain < CHAINS; ++chain) {
        sum += filler[chain];
#pragma unroll
        for (int r = 0; r < 4; ++r) sum += float(acc[chain][r]);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

#define MIXED_KERNEL(FILLERS) \
    extern "C" __global__ void peak_u8s8_mixed_##FILLERS(float* out, int iterations) { mixed_u8s8<FILLERS>(out, iterations); }
MIXED_KERNEL(0)
MIXED_KERNEL(4)
MIXED_KERNEL(8)
MIXED_KERNEL(12)
MIXED_KERNEL(16)

// 8 fillers per mma of each other kind: which pipes take time from the mma, and which run beside it.
#define MIXED_KIND_KERNEL(KIND) \
    extern "C" __global__ void peak_u8s8_mixed_8_##KIND(float* out, int iterations) { mixed_u8s8<8, KIND>(out, iterations); }
MIXED_KIND_KERNEL(FFMA_IMM)
MIXED_KIND_KERNEL(FMUL)
MIXED_KIND_KERNEL(FMNMX)
MIXED_KIND_KERNEL(IADD)
MIXED_KIND_KERNEL(I2F)
MIXED_KIND_KERNEL(EX2)
