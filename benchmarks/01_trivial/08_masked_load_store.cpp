#include <benchmark/benchmark.h>
#include <bitset>
#include <cstdint>
#include <stdio.h>

#include "../common.h"
#include "08_masked_load_store_ispc.h"

static Docs
    docs("Check masked load/store implementation.\n"
         "ISPC has several approaches for implementations of masked_load/masked_store functionality:\n"
         "1. generic implementation using per lane traversing\n"
         "2. generic implementation using llvm.masked.load/store intrinsics\n"
         "3. target-specific implementation using target-specific instructions (starting avx)\n"
         "This benchmark allows to effectively compare and tune different implementations for different types.\n"
         "Expectation:\n"
         " - No regressions\n");

WARM_UP_RUN();

// Minimum size is maximum target width, i.e. 64.
// Larger buffer is better, but preferably to stay within L1.
#define ARGS Arg(8192)
// #define ARGS RangeMultiplier(2)->Range(64, 64<<15)->Complexity(benchmark::oN)

template <typename T> static void init(T *src_a, T *src_b, T *dst, int count) {
    for (int i = 0; i < count; i++) {
        src_a[i] = static_cast<T>(i / count);
        src_b[i] = static_cast<T>((i + 1) / count);
        dst[i] = 0;
    }
}

static void init_mask_all_on(bool *mask, int count) {
    for (int i = 0; i < count; i++) {
        mask[i] = 1;
    }
}

static void init_mask_half_on(bool *mask, int count) {
    for (int i = 0; i < count; i++) {
        mask[i] = i % 2;
    }
}

template <typename T> static void check(T *src_a, T *src_b, T *dst, bool *mask, int count) {
    for (int i = 0; i < count; i++) {
        if (mask[i] == 1) {
            if (dst[i] != src_b[i] - src_a[i]) {
                printf("Error i=%d\n", i);
                return;
            }
        } else {
            if (dst[i] != 0) {
                printf("Error kuku i=%d\n", i);
                return;
            }
        }
    }
}

#define LOAD_STORE_ALL_ON(T_C, T_ISPC)                                                                                 \
    static void masked_load_store_all_on_##T_C(benchmark::State &state) {                                              \
        int count = static_cast<int>(state.range(0));                                                                  \
        T_C *src_a = static_cast<T_C *>(aligned_alloc_helper(sizeof(T_C) * count));                                    \
        T_C *src_b = static_cast<T_C *>(aligned_alloc_helper(sizeof(T_C) * count));                                    \
        T_C *dst = static_cast<T_C *>(aligned_alloc_helper(sizeof(T_C) * count));                                      \
        bool *mask = static_cast<bool *>(aligned_alloc_helper(sizeof(bool) * count));                                  \
        init(src_a, src_b, dst, count);                                                                                \
        init_mask_all_on(mask, count);                                                                                 \
        for (auto _ : state) {                                                                                         \
            ispc::masked_load_store_##T_ISPC(src_a, src_b, dst, count, mask);                                          \
        }                                                                                                              \
                                                                                                                       \
        check(src_a, src_b, dst, mask, count);                                                                         \
        aligned_free_helper(src_a);                                                                                    \
        aligned_free_helper(src_b);                                                                                    \
        aligned_free_helper(dst);                                                                                      \
        aligned_free_helper(mask);                                                                                     \
        state.SetComplexityN(state.range(0));                                                                          \
    }                                                                                                                  \
    BENCHMARK(masked_load_store_all_on_##T_C)->ARGS;

#define LOAD_STORE_HALF_ON(T_C, T_ISPC)                                                                                \
    static void masked_load_store_half_on_##T_C(benchmark::State &state) {                                             \
        int count = static_cast<int>(state.range(0));                                                                  \
        T_C *src_a = static_cast<T_C *>(aligned_alloc_helper(sizeof(T_C) * count));                                    \
        T_C *src_b = static_cast<T_C *>(aligned_alloc_helper(sizeof(T_C) * count));                                    \
        T_C *dst = static_cast<T_C *>(aligned_alloc_helper(sizeof(T_C) * count));                                      \
        bool *mask = static_cast<bool *>(aligned_alloc_helper(sizeof(bool) * count));                                  \
        init(src_a, src_b, dst, count);                                                                                \
        init_mask_half_on(mask, count);                                                                                \
        for (auto _ : state) {                                                                                         \
            ispc::masked_load_store_##T_ISPC(src_a, src_b, dst, count, mask);                                          \
        }                                                                                                              \
                                                                                                                       \
        check(src_a, src_b, dst, mask, count);                                                                         \
        aligned_free_helper(src_a);                                                                                    \
        aligned_free_helper(src_b);                                                                                    \
        aligned_free_helper(dst);                                                                                      \
        aligned_free_helper(mask);                                                                                     \
        state.SetComplexityN(state.range(0));                                                                          \
    }                                                                                                                  \
    BENCHMARK(masked_load_store_half_on_##T_C)->ARGS;

LOAD_STORE_ALL_ON(int8_t, int8)
LOAD_STORE_ALL_ON(int16_t, int16)
LOAD_STORE_ALL_ON(int32_t, int32)
LOAD_STORE_ALL_ON(float, float)
LOAD_STORE_ALL_ON(double, double)
LOAD_STORE_ALL_ON(int64_t, int64)

LOAD_STORE_HALF_ON(int8_t, int8)
LOAD_STORE_HALF_ON(int16_t, int16)
LOAD_STORE_HALF_ON(int32_t, int32)
LOAD_STORE_HALF_ON(float, float)
LOAD_STORE_HALF_ON(double, double)
LOAD_STORE_HALF_ON(int64_t, int64)

BENCHMARK_MAIN();
