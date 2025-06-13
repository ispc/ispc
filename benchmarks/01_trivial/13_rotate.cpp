// Copyright (c) 2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdint>
#include <stdio.h>

#include "../common.h"
#include "13_rotate_ispc.h"

static Docs docs("rotate_const_id - test basic rotate functionality with offset -1\n"
                 "rotate_<type> - test rotate functionality for different data types:\n"
                 "  int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float, double\n"
                 "Expectations:\n"
                 " - Each program instance should receive the value from neighbor offset steps away\n"
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

// Check function for rotate operation
template <typename T> static void check_rotate(T *dst, T *src, int count, int offset) {
    int width = ispc::width();
    for (int i = 0; i < count; i++) {
        int chunk_start = (i / width) * width;
        int lane = i % width;
        // Calculate expected lane after rotation, with proper wrapping
        int src_lane = (lane + offset + width) % width;
        int expected_index = chunk_start + src_lane;
        if (expected_index < count) {
            T expected = src[expected_index];
            if (dst[i] != expected) {
                printf("Error at i=%d: expected %d, got %d (offset=%d, lane=%d, src_lane=%d, expected_index=%d)\n", i,
                       static_cast<int>(expected), static_cast<int>(dst[i]), offset, lane, src_lane, expected_index);
                return;
            }
        }
    }
}

static void rotate_const_id(benchmark::State &state) {
    int count = static_cast<int>(state.range(0));
    int *src = static_cast<int *>(aligned_alloc_helper(sizeof(int) * count));
    int *dst = static_cast<int *>(aligned_alloc_helper(sizeof(int) * count));

    init_input(src, count);
    init_output(dst, count);

    for (auto _ : state) {
        ispc::rotate_const_id(src, dst, count);
    }

    check_rotate(dst, src, count, -1);
    aligned_free_helper(src);
    aligned_free_helper(dst);
}
BENCHMARK(rotate_const_id)->ARGS;

// Type-specific rotate benchmarks
#define ROTATE_TYPE_BENCHMARK(TYPE, TYPE_NAME)                                                                         \
    static void rotate_##TYPE_NAME(benchmark::State &state) {                                                          \
        int count = static_cast<int>(state.range(0));                                                                  \
        int offset = -1;                                                                                               \
        TYPE *src = static_cast<TYPE *>(aligned_alloc_helper(sizeof(TYPE) * count));                                   \
        TYPE *dst = static_cast<TYPE *>(aligned_alloc_helper(sizeof(TYPE) * count));                                   \
                                                                                                                       \
        init_input(src, count);                                                                                        \
        init_output(dst, count);                                                                                       \
                                                                                                                       \
        for (auto _ : state) {                                                                                         \
            ispc::rotate_##TYPE_NAME(src, dst, count, offset);                                                         \
        }                                                                                                              \
                                                                                                                       \
        check_rotate(dst, src, count, offset);                                                                         \
        aligned_free_helper(src);                                                                                      \
        aligned_free_helper(dst);                                                                                      \
    }                                                                                                                  \
    BENCHMARK(rotate_##TYPE_NAME)->ARGS;

ROTATE_TYPE_BENCHMARK(int8_t, int8)
ROTATE_TYPE_BENCHMARK(int16_t, int16)
ROTATE_TYPE_BENCHMARK(int32_t, int32)
ROTATE_TYPE_BENCHMARK(int64_t, int64)
ROTATE_TYPE_BENCHMARK(uint8_t, uint8)
ROTATE_TYPE_BENCHMARK(uint16_t, uint16)
ROTATE_TYPE_BENCHMARK(uint32_t, uint32)
ROTATE_TYPE_BENCHMARK(uint64_t, uint64)

#ifdef HAS_FP16
// Note: float16 requires special handling - using __fp16 type
static void rotate_float16(benchmark::State &state) {
    int count = static_cast<int>(state.range(0));
    int offset = -1;
    __fp16 *src = static_cast<__fp16 *>(aligned_alloc_helper(sizeof(__fp16) * count));
    __fp16 *dst = static_cast<__fp16 *>(aligned_alloc_helper(sizeof(__fp16) * count));

    // Initialize with simple values that can be represented in fp16
    for (int i = 0; i < count; i++) {
        src[i] = static_cast<__fp16>(i + 1);
        dst[i] = static_cast<__fp16>(0);
    }

    for (auto _ : state) {
        ispc::rotate_float16(src, dst, count, offset);
    }

    // Simple validation for fp16
    int width = ispc::width();
    for (int i = 0; i < count; i++) {
        int chunk_start = (i / width) * width;
        int lane = i % width;
        int src_lane = (lane + offset + width) % width;
        int expected_index = chunk_start + src_lane;
        if (expected_index < count) {
            __fp16 expected = src[expected_index];
            if (dst[i] != expected) {
                printf("Error at i=%d: expected %f, got %f (offset=%d)\n", i, static_cast<float>(expected),
                       static_cast<float>(dst[i]), offset);
                break;
            }
        }
    }

    aligned_free_helper(src);
    aligned_free_helper(dst);
}
BENCHMARK(rotate_float16)->ARGS;
#endif // HAS_FP16

ROTATE_TYPE_BENCHMARK(float, float)
ROTATE_TYPE_BENCHMARK(double, double)

BENCHMARK_MAIN();
