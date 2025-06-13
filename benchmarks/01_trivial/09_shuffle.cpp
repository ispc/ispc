// Copyright (c) 2024-2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "../common.h"
#include "09_shuffle_ispc.h"
#include <benchmark/benchmark.h>
#include <bitset>
#include <cstdint>
#include <stdio.h>

// Documentation for the benchmark
static Docs
    docs("Check performance of shuffle operations with non-constant indexes.\n"
         "ISPC has two shuffle operations:\n"
         "1. shuffle(T, int)\n"
         "2. shuffle(T, T, int)\n"
         "On some targets and for some types, shuffle operations are implemented with target-specific instructions; on "
         "others, it is just a generic implementation.\n"
         "This benchmark allows us to effectively compare and tune different implementations for different types.\n"
         "Expectation:\n"
         " - No regressions\n");

// Warm-up run for benchmarking
WARM_UP_RUN();

// Benchmark arguments
// Minimum size is maximum target width, i.e., 64. Larger buffer is better, but preferably stay within L1 cache.
#define ARGS RangeMultiplier(2)->Range(64, 64 << 10)->Complexity(benchmark::oN)

constexpr int permutation = 1;

// Initialization function for single vector shuffle
template <typename T> static void init1(T *src_a, T *dst, int count) {
    for (int i = 0; i < count; ++i) {
        src_a[i] = static_cast<T>(i);
        dst[i] = 0;
    }
}

// Initialization function for two vectors shuffle
template <typename T> static void init2(T *src_a, T *src_b, T *dst, int count) {
    for (int i = 0; i < count; ++i) {
        src_a[i] = static_cast<T>(i);
        src_b[i] = static_cast<T>(i);
        dst[i] = 0;
    }
}

// Check function for single vector shuffle
template <typename T> static void check1(T *src_a, T *dst, int count) {
    for (int i = 0; i < count; ++i) {
        if (dst[i] != src_a[i]) {
            if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, int16_t>::value ||
                          std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value) {
                printf("Error at i=%d: dst[%i]=%lld, but expected: %lld\n", i, i, (long long)dst[i],
                       (long long)src_a[i]);
#ifdef HAS_FP16
            } else if constexpr (std::is_same<T, __fp16>::value) {
                printf("Error at i=%d: dst[%i]=%f, but expected: %f\n", i, i, (float)dst[i], (float)src_a[i]);
#endif
            } else if constexpr (std::is_same<T, float>::value) {
                printf("Error at i=%d: dst[%i]=%f, but expected: %f\n", i, i, dst[i], src_a[i]);
            } else if constexpr (std::is_same<T, double>::value) {
                printf("Error at i=%d: dst[%i]=%lf, but expected: %lf\n", i, i, dst[i], src_a[i]);
            }
            return;
        }
    }
}

// Check function for two vector shuffle
template <typename T> static void check2(T *src_a, T *src_b, T *dst, int count) {
    for (int i = 0; i < count; ++i) {
        if (dst[i] != src_a[i]) {
            if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, int16_t>::value ||
                          std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value) {
                printf("Error at i=%d: dst[%i]=%lld, but expected: %lld\n", i, i, (long long)dst[i],
                       (long long)src_a[i]);
#ifdef HAS_FP16
            } else if constexpr (std::is_same<T, __fp16>::value) {
                printf("Error at i=%d: dst[%i]=%f, but expected: %f\n", i, i, (float)dst[i], (float)src_a[i]);
#endif
            } else if constexpr (std::is_same<T, float>::value) {
                printf("Error at i=%d: dst[%i]=%f, but expected: %f\n", i, i, dst[i], src_a[i]);
            } else if constexpr (std::is_same<T, double>::value) {
                printf("Error at i=%d: dst[%i]=%lf, but expected: %lf\n", i, i, dst[i], src_a[i]);
            }
            return;
        }
    }
}

// Macro to define benchmark functions for single vector shuffle
#define SHUFFLE_1(T_C, T_ISPC)                                                                                         \
    static void shuffle1_##T_C(benchmark::State &state) {                                                              \
        int count = static_cast<int>(state.range(0));                                                                  \
        T_C *src_a = static_cast<T_C *>(aligned_alloc_helper(sizeof(T_C) * count));                                    \
        T_C *dst = static_cast<T_C *>(aligned_alloc_helper(sizeof(T_C) * count));                                      \
        init1(src_a, dst, count);                                                                                      \
        for (auto _ : state) {                                                                                         \
            ispc::Shuffle1_##T_ISPC(src_a, dst, permutation, count);                                                   \
        }                                                                                                              \
        check1(src_a, dst, count);                                                                                     \
        aligned_free_helper(src_a);                                                                                    \
        aligned_free_helper(dst);                                                                                      \
        state.SetComplexityN(state.range(0));                                                                          \
    }                                                                                                                  \
    BENCHMARK(shuffle1_##T_C)->ARGS;

SHUFFLE_1(int8_t, int8)
SHUFFLE_1(int16_t, int16)
SHUFFLE_1(int, int)
#ifdef HAS_FP16
SHUFFLE_1(__fp16, float16)
#endif
SHUFFLE_1(float, float)
SHUFFLE_1(double, double)
SHUFFLE_1(int64_t, int64)

// Macro to define benchmark functions for two vector shuffle
#define SHUFFLE_2(T_C, T_ISPC)                                                                                         \
    static void shuffle2_##T_C(benchmark::State &state) {                                                              \
        int count = static_cast<int>(state.range(0));                                                                  \
        T_C *src_a = static_cast<T_C *>(aligned_alloc_helper(sizeof(T_C) * count));                                    \
        T_C *src_b = static_cast<T_C *>(aligned_alloc_helper(sizeof(T_C) * count));                                    \
        T_C *dst = static_cast<T_C *>(aligned_alloc_helper(sizeof(T_C) * count));                                      \
        init2(src_a, src_b, dst, count);                                                                               \
        for (auto _ : state) {                                                                                         \
            ispc::Shuffle2_##T_ISPC(src_a, src_b, dst, permutation, count);                                            \
        }                                                                                                              \
        check2(src_a, src_b, dst, count);                                                                              \
        aligned_free_helper(src_a);                                                                                    \
        aligned_free_helper(src_b);                                                                                    \
        aligned_free_helper(dst);                                                                                      \
        state.SetComplexityN(state.range(0));                                                                          \
    }                                                                                                                  \
    BENCHMARK(shuffle2_##T_C)->ARGS;

SHUFFLE_2(int8_t, int8)
SHUFFLE_2(int16_t, int16)
SHUFFLE_2(int, int)
#ifdef HAS_FP16
SHUFFLE_2(__fp16, float16)
#endif
SHUFFLE_2(float, float)
SHUFFLE_2(double, double)
SHUFFLE_2(int64_t, int64)

BENCHMARK_MAIN();
