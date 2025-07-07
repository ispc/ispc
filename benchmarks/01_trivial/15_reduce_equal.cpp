// Copyright (c) 2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <benchmark/benchmark.h>
#include <cstdint>
#include <stdio.h>

#include "../common.h"
#include "15_reduce_equal_ispc.h"
#include "config.h"

static Docs docs("Check reduce_equal implementation performance:\n"
                 "Tests different data types with varying input patterns.\n"
                 "Observations:\n"
                 " - reduce_equal checks if all program instances have the same value\n"
                 " - Performance depends on target architecture's reduction capabilities\n"
                 "Expectation:\n"
                 " - No regressions across data types\n");

WARM_UP_RUN();

#define ARGS Arg(256)->Arg(256 << 4)->Arg(256 << 7)->Arg(256 << 12)

template <typename T> static void init_data(T *src, int count) {
    // Same values for first half
    for (int i = 0; i < count / 2; i++) {
        src[i] = static_cast<T>(42);
    }
    // Different values for second half
    for (int i = count / 2; i < count; i++) {
        src[i] = static_cast<T>(i % 256);
    }
}

template <typename T>
static void verify_reduce_equal(T *src, bool *dst, int count, int programCount, const char *typeName) {
    // Verify each result
    for (int i = 0; i < count; i += programCount) {
        T firstValue = src[i];
        bool expected = true;

        // Check if all values in this gang are equal
        for (int j = 1; j < programCount && (i + j) < count; j++) {
            if (src[i + j] != firstValue) {
                expected = false;
                break;
            }
        }

        // Check if ISPC result matches expected result
        for (int j = 0; j < programCount && (i + j) < count; j++) {
            if (dst[i + j] != expected) {
                printf("ERROR: %s reduce_equal mismatch at index %d: expected %s, got %s\n", typeName, i + j,
                       expected ? "true" : "false", dst[i + j] ? "true" : "false");
                return;
            }
        }
    }
}

#define REDUCE_EQUAL_BENCHMARK(T_C, T_ISPC)                                                                            \
    static void reduce_equal_##T_ISPC(benchmark::State &state) {                                                       \
        int count = static_cast<int>(state.range(0));                                                                  \
        T_C *src = static_cast<T_C *>(aligned_alloc_helper(sizeof(T_C) * count));                                      \
        bool *dst = static_cast<bool *>(aligned_alloc_helper(sizeof(bool) * count));                                   \
        init_data(src, count);                                                                                         \
                                                                                                                       \
        for (auto _ : state) {                                                                                         \
            ispc::reduce_equal_##T_ISPC(src, dst, count);                                                              \
        }                                                                                                              \
                                                                                                                       \
        /* Verify correctness after benchmarking */                                                                    \
        ispc::reduce_equal_##T_ISPC(src, dst, count);                                                                  \
        verify_reduce_equal(src, dst, count, ispc::width(), #T_ISPC);                                                  \
                                                                                                                       \
        aligned_free_helper(src);                                                                                      \
        aligned_free_helper(dst);                                                                                      \
        state.SetComplexityN(state.range(0));                                                                          \
    }                                                                                                                  \
    BENCHMARK(reduce_equal_##T_ISPC)->ARGS;

// Generate benchmarks for each type
REDUCE_EQUAL_BENCHMARK(int32_t, int32)
REDUCE_EQUAL_BENCHMARK(int64_t, int64)
REDUCE_EQUAL_BENCHMARK(float, float)
REDUCE_EQUAL_BENCHMARK(double, double)
REDUCE_EQUAL_BENCHMARK(int8_t, int8)
REDUCE_EQUAL_BENCHMARK(int16_t, int16)
#ifdef HAS_FP16
REDUCE_EQUAL_BENCHMARK(__fp16, float16)
#endif

BENCHMARK_MAIN();
