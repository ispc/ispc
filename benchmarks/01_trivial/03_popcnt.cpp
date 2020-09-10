#include <benchmark/benchmark.h>
#include <bitset>
#include <cstdint>
#include <stdio.h>

#include "../common.h"
#include "03_popcnt_ispc.h"

static Docs docs("Check popcnt implmentation of stdlib functions:\n"
                 "[int32, int64] x [uniform, varying] x [all, even] versions.\n"
                 "Observations:\n"
                 " - popcnt is very lightweight, so 8 popcnt are chained.\n"
                 " - chaining doesn't cause optimizing out popcnt, 8 seems to be good enough"
                 " - varying versions have overhead on insert/extract\n"
                 " - for even versions mask is statically known and compiler is able to optimize out inactive lanes\n"
                 "Expectation:\n"
                 " - No regressions\n");

// Minimum size is maximum target width, i.e. 64.
// Larger buffer is better, but preferably to stay within L1.
#define ARGS Arg(8192)
//#define ARGS RangeMultiplier(2)->Range(64, 64<<15)->Complexity(benchmark::oN)

template <typename T> static void init(T *src, T *dst, int count) {
    for (int i = 0; i < count; i++) {
        src[i] = static_cast<T>(i);
        dst[i] = 0;
    }
}

template <typename T> static void check_all(T *src, T *dst, int count) {
    for (int i = 0; i < count; i++) {
        // for single popcnt() here's the formula, but we chain 8 popcnt()
        // int count = std::bitset< std::numeric_limits<T>::digits >(src[i]).count();
        int count = (i == 0) ? 0 : 1;
        if (dst[i] != count) {
            printf("Error i=%d\n", i);
            return;
        }
    }
}

template <typename T> static void check_even(T *src, T *dst, int count) {
    for (int i = 0; i < count; i += 2) {
        // for single popcnt() here's the formula, but we chain 8 popcnt()
        // int count = std::bitset< std::numeric_limits<T>::digits >(src[i]).count();
        int count = (i == 0) ? 0 : 1;
        if (dst[i] != count) {
            printf("Error i=%d\n", i);
            return;
        }
    }
}

#define POPCNT(T_C, T_ISPC, V, ALL)                                                                                    \
    static void popcnt_stdlib_##V##_##T_ISPC##_##ALL(benchmark::State &state) {                                        \
        int count = static_cast<int>(state.range(0));                                                                  \
        T_C *src = static_cast<T_C *>(aligned_alloc_helper(sizeof(T_C) * count));                                      \
        T_C *dst = static_cast<T_C *>(aligned_alloc_helper(sizeof(T_C) * count));                                      \
        init(src, dst, count);                                                                                         \
                                                                                                                       \
        for (auto _ : state) {                                                                                         \
            ispc::popcnt_##V##_##T_ISPC##_##ALL(src, dst, count);                                                      \
        }                                                                                                              \
                                                                                                                       \
        check_##ALL(src, dst, count);                                                                                  \
        aligned_free_helper(src);                                                                                      \
        aligned_free_helper(dst);                                                                                      \
        state.SetComplexityN(state.range(0));                                                                          \
    }                                                                                                                  \
    BENCHMARK(popcnt_stdlib_##V##_##T_ISPC##_##ALL)->ARGS;

POPCNT(int, int32, uniform, all)
POPCNT(int, int32, varying, all)
POPCNT(int, int32, uniform, even)
POPCNT(int, int32, varying, even)
POPCNT(int64_t, int64, uniform, all)
POPCNT(int64_t, int64, varying, all)
POPCNT(int64_t, int64, uniform, even)
POPCNT(int64_t, int64, varying, even)

BENCHMARK_MAIN();
