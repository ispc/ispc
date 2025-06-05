// Copyright (c) 2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdint>
#include <stdio.h>

#include "../common.h"
#include "12_broadcast_ispc.h"

static Docs docs("broadcast_varying_index - test broadcast from different program indices\n"
                 "broadcast_with_computation - test broadcast with computation before broadcasting\n"
                 "broadcast_<type> - test broadcast functionality for different data types:\n"
                 "  int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float, double\n"
                 "Expectations:\n"
                 " - All program instances should receive the same value from the broadcasting instance\n"
                 " - Performance should be consistent across different data types\n");

WARM_UP_RUN();

// Use standard argument sizes for consistency with other benchmarks
#define ARGS Arg(256)->Arg(256 << 4)->Arg(256 << 7)->Arg(256 << 12)

// Helper functions for different data types
template <typename T> static void init_input(T *src, int count) {
    for (int i = 0; i < count; i++) {
        src[i] = static_cast<T>(i + 1);
    }
}

template <typename T> static void init_output(T *dst, int count) {
    for (int i = 0; i < count; i++) {
        dst[i] = static_cast<T>(0);
    }
}

// Check function for simple broadcast (should all be value from index 0)
template <typename T> static void check_simple_broadcast(T *dst, T *src, int count) {
    int width = ispc::width();
    for (int i = 0; i < count; i++) {
        int chunk_start = (i / width) * width;
        T expected = src[chunk_start]; // Value from lane 0 of each chunk
        if (dst[i] != expected) {
            printf("Error at i=%d: expected %d, got %d (chunk_start=%d)\n", i, static_cast<int>(expected),
                   static_cast<int>(dst[i]), chunk_start);
            return;
        }
    }
}

// Check function for broadcast with computation
static void check_broadcast_computation(float *dst, float *src, int count) {
    const float rel_epsilon = 1e-5f; // Relative tolerance for floating-point comparison
    int width = ispc::width();
    for (int i = 0; i < count; i++) {
        int chunk_start = (i / width) * width;
        float expected = src[chunk_start] * src[chunk_start] + 1.0f; // Expected computation from lane 0 of each chunk
        float actual = dst[i];
        float tolerance = rel_epsilon * std::abs(expected);
        if (std::abs(actual - expected) > tolerance) {
            printf("Error at i=%d: expected %f, got %f, diff=%f, tolerance=%f (chunk_start=%d)\n", i, expected, actual,
                   std::abs(actual - expected), tolerance, chunk_start);
            return;
        }
    }
}

static void broadcast_varying_index(benchmark::State &state) {
    int count = static_cast<int>(state.range(0));
    int width = ispc::width();
    int src_index = 1; // Test broadcasting from lane 1 (if available)
    if (src_index >= width)
        src_index = 0;

    int *src = static_cast<int *>(aligned_alloc_helper(sizeof(int) * count));
    int *dst = static_cast<int *>(aligned_alloc_helper(sizeof(int) * count));

    init_input(src, count);
    init_output(dst, count);

    for (auto _ : state) {
        ispc::broadcast_varying_index(src, dst, count, src_index);
    }

    // Validate that broadcast worked - should get value from src_index lane of each chunk
    for (int i = 0; i < count; i++) {
        int chunk_start = (i / width) * width;
        int expected_index = chunk_start + src_index;
        if (expected_index < count) {
            int expected = src[expected_index];
            if (dst[i] != expected) {
                printf("Error at i=%d: expected %d, got %d (from src[%d], chunk_start=%d)\n", i, expected, dst[i],
                       expected_index, chunk_start);
                break;
            }
        }
    }

    aligned_free_helper(src);
    aligned_free_helper(dst);
}
BENCHMARK(broadcast_varying_index)->ARGS;

static void broadcast_with_computation(benchmark::State &state) {
    int count = static_cast<int>(state.range(0));
    float *src = static_cast<float *>(aligned_alloc_helper(sizeof(float) * count));
    float *dst = static_cast<float *>(aligned_alloc_helper(sizeof(float) * count));

    init_input(src, count);
    init_output(dst, count);

    for (auto _ : state) {
        ispc::broadcast_with_computation(src, dst, count);
    }

    check_broadcast_computation(dst, src, count);
    aligned_free_helper(src);
    aligned_free_helper(dst);
}
BENCHMARK(broadcast_with_computation)->ARGS;

// Type-specific broadcast benchmarks
#define BROADCAST_TYPE_BENCHMARK(TYPE, TYPE_NAME)                                                                      \
    static void broadcast_##TYPE_NAME(benchmark::State &state) {                                                       \
        int count = static_cast<int>(state.range(0));                                                                  \
        TYPE *src = static_cast<TYPE *>(aligned_alloc_helper(sizeof(TYPE) * count));                                   \
        TYPE *dst = static_cast<TYPE *>(aligned_alloc_helper(sizeof(TYPE) * count));                                   \
                                                                                                                       \
        init_input(src, count);                                                                                        \
        init_output(dst, count);                                                                                       \
                                                                                                                       \
        for (auto _ : state) {                                                                                         \
            ispc::broadcast_##TYPE_NAME(src, dst, count);                                                              \
        }                                                                                                              \
                                                                                                                       \
        check_simple_broadcast(dst, src, count);                                                                       \
        aligned_free_helper(src);                                                                                      \
        aligned_free_helper(dst);                                                                                      \
    }                                                                                                                  \
    BENCHMARK(broadcast_##TYPE_NAME)->ARGS;

BROADCAST_TYPE_BENCHMARK(int8_t, int8)
BROADCAST_TYPE_BENCHMARK(int16_t, int16)
BROADCAST_TYPE_BENCHMARK(int32_t, int32)
BROADCAST_TYPE_BENCHMARK(int64_t, int64)
BROADCAST_TYPE_BENCHMARK(uint8_t, uint8)
BROADCAST_TYPE_BENCHMARK(uint16_t, uint16)
BROADCAST_TYPE_BENCHMARK(uint32_t, uint32)
BROADCAST_TYPE_BENCHMARK(uint64_t, uint64)
#ifdef HAS_FP16
// Note: float16 requires special handling - using __fp16 type
static void broadcast_float16(benchmark::State &state) {
    int count = static_cast<int>(state.range(0));
    __fp16 *src = static_cast<__fp16 *>(aligned_alloc_helper(sizeof(__fp16) * count));
    __fp16 *dst = static_cast<__fp16 *>(aligned_alloc_helper(sizeof(__fp16) * count));

    // Initialize with simple values that can be represented in fp16
    for (int i = 0; i < count; i++) {
        src[i] = static_cast<__fp16>(i + 1);
        dst[i] = static_cast<__fp16>(0);
    }

    for (auto _ : state) {
        ispc::broadcast_float16(src, dst, count);
    }

    // Simple validation for fp16
    int width = ispc::width();
    for (int i = 0; i < count; i++) {
        int chunk_start = (i / width) * width;
        __fp16 expected = src[chunk_start];
        if (dst[i] != expected) {
            printf("Error at i=%d: expected %f, got %f (chunk_start=%d)\n", i, static_cast<float>(expected),
                   static_cast<float>(dst[i]), chunk_start);
            break;
        }
    }

    aligned_free_helper(src);
    aligned_free_helper(dst);
}
BENCHMARK(broadcast_float16)->ARGS;
#endif // HAS_FP16

BROADCAST_TYPE_BENCHMARK(float, float)
BROADCAST_TYPE_BENCHMARK(double, double)

BENCHMARK_MAIN();
