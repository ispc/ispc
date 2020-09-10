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
                 "Expectation:\n"
                 " - No regressions\n");

// Minimum size is maximum target width, i.e. 64.
// Larger buffer is better, but preferably to stay within L1.
#define ARGS Arg(8192)
//#define ARGS RangeMultiplier(2)->Range(64, 64<<15)->Complexity(benchmark::oN)

template <typename T> static void init_src(T *src, int count) {
    for (int i = 0; i < count; i++) {
        // These computations may involve overflow/underflow, but this is ok.
        src[i] = ((T)i) - ((T)count / 2);
    }
}

template <typename T> static void init_dst(T *dst, int count) {
    for (int i = 0; i < count; i++) {
        dst[i] = 0;
    }
}

template <typename T> static void check(T *src, T *dst, int divisor, int count) {
    for (int i = 0; i < count; i++) {
        T val = src[i] / divisor;
        if (val != dst[i]) {
            printf("Error i=%d\n", i);
            return;
        }
    }
}

#define FASTDIV(T_C, T_ISPC, DIV_VAL)                                                                                  \
    static void fastdiv_##T_ISPC##_##DIV_VAL(benchmark::State &state) {                                                \
        int count = static_cast<int>(state.range(0));                                                                  \
        T_C *dst = static_cast<T_C *>(aligned_alloc_helper(sizeof(T_C) * count));                                      \
        T_C *src = static_cast<T_C *>(aligned_alloc_helper(sizeof(T_C) * count));                                      \
        init_src(src, count);                                                                                          \
        init_dst(dst, count);                                                                                          \
                                                                                                                       \
        for (auto _ : state) {                                                                                         \
            ispc::fastdiv_##T_ISPC##_##DIV_VAL(src, dst, count);                                                       \
        }                                                                                                              \
                                                                                                                       \
        check(src, dst, DIV_VAL, count);                                                                               \
        aligned_free_helper(src);                                                                                      \
        aligned_free_helper(dst);                                                                                      \
        state.SetComplexityN(state.range(0));                                                                          \
    }                                                                                                                  \
    BENCHMARK(fastdiv_##T_ISPC##_##DIV_VAL)->ARGS;

FASTDIV(uint64_t, uint64, 13)
FASTDIV(int64_t, int64, 13)

FASTDIV(uint32_t, uint32, 13)
FASTDIV(int32_t, int32, 13)

FASTDIV(uint16_t, uint16, 13)
FASTDIV(int16_t, int16, 13)

FASTDIV(uint8_t, uint8, 13)
FASTDIV(int8_t, int8, 13)

FASTDIV(uint64_t, uint64, 16)
FASTDIV(int64_t, int64, 16)

FASTDIV(uint32_t, uint32, 16)
FASTDIV(int32_t, int32, 16)

FASTDIV(uint16_t, uint16, 16)
FASTDIV(int16_t, int16, 16)

FASTDIV(uint8_t, uint8, 16)
FASTDIV(int8_t, int8, 16)

BENCHMARK_MAIN();
