// Copyright (c) 2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdint>
#include <stdio.h>

#include "../common.h"
#include "14_shift_ispc.h"

static Docs docs("shift_const_<type> - test basic shift functionality with offset -1\n"
                 "shift_<type> - test shift functionality for different data types:\n"
                 "  int8, int16, int32, int64, float16, float, double\n"
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

// Check function for shift operation
template <typename T> static void check_shift(T *dst, T *src, int count, int offset) {
    int width = ispc::width();

    for (int i = 0; i < count; i++) {
        int chunk_start = (i / width) * width;
        int lane = i % width;

        // For shift operation: dst[lane] = src[lane + offset] or 0 if out of bounds
        // shift(x, -1) = [0,1,2,3] means lane 1 gets src[0], lane 2 gets src[1], etc.
        int src_lane = lane + offset;
        T expected = static_cast<T>(0); // Default to zero for out-of-bounds

        if (src_lane >= 0 && src_lane < width) {
            int expected_index = chunk_start + src_lane;
            if (expected_index < count) {
                expected = src[expected_index];
            }
        }

        if (dst[i] != expected) {
            printf("Error at i=%d: expected %d, got %d (offset=%d, lane=%d, src_lane=%d, width=%d)\n", i,
                   static_cast<int>(expected), static_cast<int>(dst[i]), offset, lane, src_lane, width);
            return;
        }
    }
}

// Type-specific shift benchmarks
#define SHIFT_TYPE_BENCHMARK(TYPE, TYPE_NAME)                                                                          \
    static void shift_##TYPE_NAME(benchmark::State &state) {                                                           \
        int count = static_cast<int>(state.range(0));                                                                  \
        int offset = -1;                                                                                               \
        TYPE *src = static_cast<TYPE *>(aligned_alloc_helper(sizeof(TYPE) * count));                                   \
        TYPE *dst = static_cast<TYPE *>(aligned_alloc_helper(sizeof(TYPE) * count));                                   \
                                                                                                                       \
        init_input(src, count);                                                                                        \
        init_output(dst, count);                                                                                       \
                                                                                                                       \
        for (auto _ : state) {                                                                                         \
            ispc::shift_##TYPE_NAME(src, dst, count, offset);                                                          \
        }                                                                                                              \
                                                                                                                       \
        check_shift(dst, src, count, offset);                                                                          \
        aligned_free_helper(src);                                                                                      \
        aligned_free_helper(dst);                                                                                      \
    }                                                                                                                  \
    BENCHMARK(shift_##TYPE_NAME)->ARGS;

// Const shift benchmarks with fixed offset -1
#define SHIFT_CONST_BENCHMARK(TYPE, TYPE_NAME)                                                                         \
    static void shift_const_##TYPE_NAME(benchmark::State &state) {                                                     \
        int count = static_cast<int>(state.range(0));                                                                  \
        TYPE *src = static_cast<TYPE *>(aligned_alloc_helper(sizeof(TYPE) * count));                                   \
        TYPE *dst = static_cast<TYPE *>(aligned_alloc_helper(sizeof(TYPE) * count));                                   \
                                                                                                                       \
        init_input(src, count);                                                                                        \
        init_output(dst, count);                                                                                       \
                                                                                                                       \
        for (auto _ : state) {                                                                                         \
            ispc::shift_const_##TYPE_NAME(src, dst, count);                                                            \
        }                                                                                                              \
                                                                                                                       \
        check_shift(dst, src, count, -1);                                                                              \
        aligned_free_helper(src);                                                                                      \
        aligned_free_helper(dst);                                                                                      \
    }                                                                                                                  \
    BENCHMARK(shift_const_##TYPE_NAME)->ARGS;

SHIFT_TYPE_BENCHMARK(int8_t, int8)
SHIFT_CONST_BENCHMARK(int8_t, int8)
SHIFT_TYPE_BENCHMARK(int16_t, int16)
SHIFT_CONST_BENCHMARK(int16_t, int16)
SHIFT_TYPE_BENCHMARK(int32_t, int32)
SHIFT_CONST_BENCHMARK(int32_t, int32)
SHIFT_TYPE_BENCHMARK(int64_t, int64)
SHIFT_CONST_BENCHMARK(int64_t, int64)

#ifdef HAS_FP16
// Note: float16 requires special handling - using __fp16 type
static void shift_float16(benchmark::State &state) {
    int count = static_cast<int>(state.range(0));
    int offset = -1;
    __fp16 *src = static_cast<__fp16 *>(aligned_alloc_helper(sizeof(__fp16) * count));
    __fp16 *dst = static_cast<__fp16 *>(aligned_alloc_helper(sizeof(__fp16) * count));

    // Initialize with small values that can be represented accurately in fp16
    // fp16 has limited precision, so use values that won't cause rounding errors
    for (int i = 0; i < count; i++) {
        src[i] = static_cast<__fp16>((i % 1000) + 1); // Limit to 1-1000 range
        dst[i] = static_cast<__fp16>(0);
    }

    for (auto _ : state) {
        ispc::shift_float16(src, dst, count, offset);
    }

    // Simple validation for fp16
    int width = ispc::width();
    for (int i = 0; i < count; i++) {
        int chunk_start = (i / width) * width;
        int lane = i % width;
        int src_lane = lane + offset;
        __fp16 expected = static_cast<__fp16>(0);

        if (src_lane >= 0 && src_lane < width) {
            int expected_index = chunk_start + src_lane;
            if (expected_index < count) {
                expected = src[expected_index];
            }
        }

        if (dst[i] != expected) {
            printf("Error at i=%d: expected %f, got %f (offset=%d)\n", i, static_cast<float>(expected),
                   static_cast<float>(dst[i]), offset);
            break;
        }
    }

    aligned_free_helper(src);
    aligned_free_helper(dst);
}
BENCHMARK(shift_float16)->ARGS;

// Const float16 benchmark
static void shift_const_float16(benchmark::State &state) {
    int count = static_cast<int>(state.range(0));
    __fp16 *src = static_cast<__fp16 *>(aligned_alloc_helper(sizeof(__fp16) * count));
    __fp16 *dst = static_cast<__fp16 *>(aligned_alloc_helper(sizeof(__fp16) * count));

    // Initialize with small values that can be represented accurately in fp16
    for (int i = 0; i < count; i++) {
        src[i] = static_cast<__fp16>((i % 1000) + 1);
        dst[i] = static_cast<__fp16>(0);
    }

    for (auto _ : state) {
        ispc::shift_const_float16(src, dst, count);
    }

    // Simple validation for fp16
    int width = ispc::width();
    for (int i = 0; i < count; i++) {
        int chunk_start = (i / width) * width;
        int lane = i % width;
        int src_lane = lane + (-1);
        __fp16 expected = static_cast<__fp16>(0);

        if (src_lane >= 0 && src_lane < width) {
            int expected_index = chunk_start + src_lane;
            if (expected_index < count) {
                expected = src[expected_index];
            }
        }

        if (dst[i] != expected) {
            printf("Error at i=%d: expected %f, got %f (offset=-1)\n", i, static_cast<float>(expected),
                   static_cast<float>(dst[i]));
            break;
        }
    }

    aligned_free_helper(src);
    aligned_free_helper(dst);
}
BENCHMARK(shift_const_float16)->ARGS;
#endif // HAS_FP16

SHIFT_TYPE_BENCHMARK(float, float)
SHIFT_CONST_BENCHMARK(float, float)
SHIFT_TYPE_BENCHMARK(double, double)
SHIFT_CONST_BENCHMARK(double, double)

BENCHMARK_MAIN();
