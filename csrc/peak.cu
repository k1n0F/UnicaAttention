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
