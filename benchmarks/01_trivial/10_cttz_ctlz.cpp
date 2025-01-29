// Copyright (c) 2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <benchmark/benchmark.h>
#include <bitset>
#include <climits>
#include <cstdint>
#include <stdio.h>

#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include "../common.h"
#include "10_cttz_ctlz_ispc.h"

static Docs docs("Check count_trailing_zeros and count_leading_zeros implementation of stdlib functions:\n"
                 "[int32, int64] x [uniform, varying] versions.\n"
                 "Expectation:\n"
                 " - No regressions\n");

WARM_UP_RUN();

#define ARGS Arg(8192)

// Generic initialization function
template <typename T> static void init(T *src, T *dst, int count) {
    for (int i = 0; i < count; i++) {
        src[i] = static_cast<T>(i);
        dst[i] = 0;
    }
}

template <typename T> static T count_trailing_zeros(T value) {
    if (value == 0)
        return sizeof(T) * CHAR_BIT;

#if defined(_MSC_VER)
    unsigned long index;
    if constexpr (sizeof(T) == 8) {
        _BitScanForward64(&index, value);
        return index;
    } else {
        _BitScanForward(&index, value);
        return index;
    }
#else
    if constexpr (sizeof(T) == 8) {
        return __builtin_ctzll(value);
    } else {
        return __builtin_ctz(value);
    }
#endif
}

template <typename T> static T count_leading_zeros(T value) {
    if (value == 0)
        return sizeof(T) * CHAR_BIT;

#if defined(_MSC_VER)
    unsigned long index;
    if constexpr (sizeof(T) == 8) {
        _BitScanReverse64(&index, value);
        return index;
    } else {
        _BitScanReverse(&index, value);
        return index;
    }
#else
    if constexpr (sizeof(T) == 8) {
        return __builtin_clzll(value);
    } else {
        return __builtin_clz(value);
    }
#endif
}

// Checking functions
template <typename T> static void check_cttz(const T *src, const T *dst, int count) {
    for (int i = 0; i < count; ++i) {
        T expected = count_trailing_zeros(src[i]);
        if (expected != dst[i]) {
            printf("Error i=%d expected=%d result=%d\n", i, (int)expected, (int)dst[i]);
            return;
        }
    }
}

template <typename T> static void check_ctlz(const T *src, const T *dst, int count) {
    for (int i = 0; i < count; ++i) {
        T expected = count_leading_zeros(src[i]);
        if (expected != dst[i]) {
            printf("Error i=%d expected=%d result=%d\n", i, (int)expected, (int)dst[i]);
            return;
        }
    }
}

// Benchmark macro
#define BENCHMARK_BIT_OP(OP, T_C, T_ISPC, V)                                                                           \
    static void OP##_stdlib_##V##_##T_ISPC(benchmark::State &state) {                                                  \
        int count = static_cast<int>(state.range(0));                                                                  \
        T_C *src = static_cast<T_C *>(aligned_alloc_helper(sizeof(T_C) * count));                                      \
        T_C *dst = static_cast<T_C *>(aligned_alloc_helper(sizeof(T_C) * count));                                      \
        init(src, dst, count);                                                                                         \
                                                                                                                       \
        for (auto _ : state) {                                                                                         \
            ispc::OP##_##V##_##T_ISPC(src, dst, count);                                                                \
        }                                                                                                              \
                                                                                                                       \
        check_##OP(src, dst, count);                                                                                   \
        aligned_free_helper(src);                                                                                      \
        aligned_free_helper(dst);                                                                                      \
        state.SetComplexityN(state.range(0));                                                                          \
    }                                                                                                                  \
    BENCHMARK(OP##_stdlib_##V##_##T_ISPC)->ARGS;

// Generate all benchmark combinations
BENCHMARK_BIT_OP(cttz, int, int32, uniform)
BENCHMARK_BIT_OP(cttz, int, int32, varying)
BENCHMARK_BIT_OP(cttz, int64_t, int64, uniform)
BENCHMARK_BIT_OP(cttz, int64_t, int64, varying)

BENCHMARK_BIT_OP(ctlz, int, int32, uniform)
BENCHMARK_BIT_OP(ctlz, int, int32, varying)
BENCHMARK_BIT_OP(ctlz, int64_t, int64, uniform)
BENCHMARK_BIT_OP(ctlz, int64_t, int64, varying)

BENCHMARK_MAIN();