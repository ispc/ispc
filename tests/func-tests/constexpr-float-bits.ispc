// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "test_static.isph"

// Cover negative zero, negative infinity, and quiet and signaling NaNs with payloads.
#define TEST_FLOAT_BITS(NAME, FLOAT, UINT, FROM_BITS, ZERO, INF, QNAN, SNAN, SIGN)                                      \
    constexpr uniform FLOAT fixed_##NAME[4] = {FROM_BITS((uniform UINT)ZERO), FROM_BITS((uniform UINT)INF),              \
                                                FROM_BITS((uniform UINT)QNAN), FROM_BITS((uniform UINT)SNAN)};          \
    constexpr varying UINT lane_bits_##NAME = (varying UINT)((programIndex & 2)                                         \
                                                                ? ((programIndex & 1) ? SNAN : QNAN)                   \
                                                                : ((programIndex & 1) ? INF : ZERO));                  \
    constexpr varying FLOAT lane_value_##NAME = FROM_BITS(lane_bits_##NAME);                                           \
    constexpr varying UINT lane_abs_##NAME = intbits(abs(lane_value_##NAME));                                          \
    constexpr varying UINT lane_sign_##NAME = signbits(lane_value_##NAME);                                             \
    constexpr varying bool lane_inf_##NAME = isinf(lane_value_##NAME);                                                 \
    constexpr varying bool lane_finite_##NAME = isfinite(lane_value_##NAME);                                           \
    static inline varying bool test_##NAME(uniform int offset) {                                                       \
        uniform UINT patterns[4] = {ZERO, INF, QNAN, SNAN};                                                            \
        uniform bool scalar_ok = true;                                                                                \
        for (uniform int i = 0; i < 4; ++i) {                                                                         \
            uniform int index = (i + offset) & 3;                                                                     \
            uniform UINT bits = patterns[index];                                                                      \
            uniform FLOAT runtime_value = FROM_BITS(bits);                                                           \
            scalar_ok &= intbits(fixed_##NAME[index]) == bits && intbits(runtime_value) == bits &&                     \
                         intbits(abs(runtime_value)) == (bits & ~(uniform UINT)SIGN) &&                                \
                         signbits(runtime_value) == (bits & (uniform UINT)SIGN) &&                                     \
                         isinf(runtime_value) == (index == 1) && isfinite(runtime_value) == (index == 0);              \
        }                                                                                                             \
        varying int index = (programIndex + offset) & 3;                                                              \
        varying UINT bits = patterns[index];                                                                         \
        varying FLOAT runtime_value = FROM_BITS(bits);                                                                \
        return scalar_ok && intbits(lane_value_##NAME) == lane_bits_##NAME &&                                          \
               lane_abs_##NAME == (lane_bits_##NAME & ~(varying UINT)SIGN) &&                                           \
               lane_sign_##NAME == (lane_bits_##NAME & (varying UINT)SIGN) &&                                           \
               lane_inf_##NAME == ((programIndex & 3) == 1) && lane_finite_##NAME == ((programIndex & 3) == 0) &&        \
               intbits(runtime_value) == bits && intbits(abs(runtime_value)) == (bits & ~(varying UINT)SIGN) &&        \
               signbits(runtime_value) == (bits & (varying UINT)SIGN) && isinf(runtime_value) == (index == 1) &&        \
               isfinite(runtime_value) == (index == 0);                                                                \
    }

TEST_FLOAT_BITS(half, float16, uint16, float16bits, 0x8000, 0xfc00, 0xfe35, 0xfc35, 0x8000)
TEST_FLOAT_BITS(single, float, uint32, floatbits, 0x80000000u, 0xff800000u, 0xffc00135u, 0xff800135u, 0x80000000u)
TEST_FLOAT_BITS(double, double, uint64, doublebits, 0x8000000000000000ull, 0xfff0000000000000ull,
                0xfff8000000000135ull, 0xfff0000000000135ull, 0x8000000000000000ull)
#undef TEST_FLOAT_BITS

task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int offset = aFOO[0] > 0 ? 0 : 1;
    RET[programIndex] = test_half(offset) && test_single(offset) && test_double(offset) ? 1.f : 0.f;
}

task void result(uniform float RET[]) { RET[programIndex] = 1.f; }
