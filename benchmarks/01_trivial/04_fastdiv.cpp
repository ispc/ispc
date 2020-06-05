#include <benchmark/benchmark.h>
#include <bitset>
#include <cstdint>
#include <stdio.h>

#include "../common.h"
#include "04_fastdiv_ispc.h"

static Docs docs("Check fast_idiv implmentation of stdlib functions:\n"
                 "[int8, uint8, int16, uint16, int32, uint32, int64, uint64] x [13, 16] versions.\n"
                 "Conditions to trigger fast_idiv:\n"
                 " - The value being divided must be an int8/16/32.\n"
                 " - The divisor must be the same compile-time constant value for all of the vector lanes.\n"
                 " - The divisor must be >= 2 and <128 (for 8-bit divides), and <256 otherwise.\n"
                 "Expectation:\n"
                 " - No regressions\n");

// Minimum size is maximum target width, i.e. 64.
// Larger buffer is better, but preferably to stay within L1.
// For int8, maximum value is 128.
#define ARGS_int8 Arg(128)
#define ARGS_uint8 Arg(256)
#define ARGS_int16 Arg(8192)
#define ARGS_uint16 Arg(8192)
#define ARGS_int32 Arg(8192)
#define ARGS_uint32 Arg(8192)
#define ARGS_int64 Arg(8192)
#define ARGS_uint64 Arg(8192)
//#define ARGS RangeMultiplier(2)->Range(64, 64<<15)->Complexity(benchmark::oN)

template <typename T> static void init(T *dst, int count) {
    for (int i = 0; i < count; i++) {
        dst[i] = 0;
    }
}

template <typename T> static void check_unsigned(T *dst, int divisor, int count) {
    for (int i = 0; i < count; i++) {
        T num = i;
        T val = num / divisor;
        if (val != dst[i]) {
            printf("Error i=%d\n", i);
            return;
        }
    }
}

template <typename T> static void check_signed(T *dst, int divisor, int count) {
    for (int i = 0; i < 2 * count; i++) {
        T num = i - count;
        T val = num / divisor;
        if (val != dst[i]) {
            printf("Error i=%d\n", i);
            return;
        }
    }
}

#define FASTDIV(UT_C, UT_ISPC, ST_C, ST_ISPC, DIV_VAL)                                                                 \
    static void fastdiv_##UT_ISPC##_##DIV_VAL(benchmark::State &state) {                                               \
        int count = state.range(0);                                                                                    \
        UT_C *dst = static_cast<UT_C *>(aligned_alloc(ALIGNMENT, sizeof(UT_C) * count));                               \
        init(dst, count);                                                                                              \
                                                                                                                       \
        for (auto _ : state) {                                                                                         \
            ispc::fastdiv_##UT_ISPC##_##DIV_VAL(dst, count);                                                           \
        }                                                                                                              \
                                                                                                                       \
        check_unsigned(dst, DIV_VAL, count);                                                                           \
        free(dst);                                                                                                     \
        state.SetComplexityN(state.range(0));                                                                          \
    }                                                                                                                  \
    BENCHMARK(fastdiv_##UT_ISPC##_##DIV_VAL)->ARGS_##UT_ISPC;                                                          \
                                                                                                                       \
    static void fastdiv_##ST_ISPC##_##DIV_VAL(benchmark::State &state) {                                               \
        int count = state.range(0);                                                                                    \
        ST_C *dst = static_cast<ST_C *>(aligned_alloc(ALIGNMENT, sizeof(ST_C) * 2 * count));                           \
        init(dst, 2 * count);                                                                                          \
                                                                                                                       \
        for (auto _ : state) {                                                                                         \
            ispc::fastdiv_##ST_ISPC##_##DIV_VAL(dst, count);                                                           \
        }                                                                                                              \
                                                                                                                       \
        check_signed(dst, DIV_VAL, count);                                                                             \
        free(dst);                                                                                                     \
        state.SetComplexityN(state.range(0));                                                                          \
    }                                                                                                                  \
    BENCHMARK(fastdiv_##ST_ISPC##_##DIV_VAL)->ARGS_##ST_ISPC;

FASTDIV(uint64_t, uint64, int64_t, int64, 13)

FASTDIV(uint32_t, uint32, int32_t, int32, 13)

FASTDIV(uint16_t, uint16, int16_t, int16, 13)

FASTDIV(uint8_t, uint8, int8_t, int8, 13)

FASTDIV(uint64_t, uint64, int64_t, int64, 16)

FASTDIV(uint32_t, uint32, int32_t, int32, 16)

FASTDIV(uint16_t, uint16, int16_t, int16, 16)

FASTDIV(uint8_t, uint8, int8_t, int8, 16)

BENCHMARK_MAIN();
