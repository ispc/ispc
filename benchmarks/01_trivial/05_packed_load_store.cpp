#include <benchmark/benchmark.h>
#include <bitset>
#include <cstdint>
#include <stdio.h>

#include "../common.h"
#include "05_packed_load_store_ispc.h"

static Docs
    docs("Check packed_load_active/packed_store_active implementation of stdlib functions:\n"
         "[int32, int64] x [all_off, 1/16, 1/8, 1/4, 1/2, 3/4, 7/8, 15/16, all_on] versions.\n"
         "Observation:\n"
         " - it does make sense to test multiple mask pattern, as they behave substantantially differently for "
         "different implementations\n"
         " - implementations based on LLVM intrinsic is a good one, but not the best, as it don't do popcnt as "
         "part of intrinsic\n"
         " - performance was not evaluated for different data set sizes\n"
         " - 1/16, 1/8 behave like 1/4 for 4 wide targets, similarly 1/16 on 8 wide is 1/8\n"
         " - 1/2 even and 1/2 odd have different perfomance, that's not expected.\n"
         " - 1/2 even is about 20% regression on AVX2 (LLVM intrinsics vs old manual implementation) that's might be "
         "interesting to investigate\n"
         "Expectation:\n"
         " - No regressions\n");

// Minimum size is maximum target width, i.e. 64.
// Larger buffer is better, but preferably to stay within L1.
#define ARGS Arg(8192)
//#define ARGS RangeMultiplier(2)->Range(64, 64<<15)->Complexity(benchmark::oN)

template <typename T> static void init(T *src, T *dst, int count) {
    for (int i = 0; i < count; i++) {
        src[i] = static_cast<T>(i + 1);
        dst[i] = 0;
    }
}

template <typename T>
static void check_packed_load_active_eq(T *src, T *dst, const unsigned int index, unsigned int num, int count) {
    int width = ispc::width();
    int loc = 0;
    int checkVal = 0;
    for (int i = 0; i < count / width; i++) {
        for (int programIndex = 0; programIndex < width; programIndex++) {
            if ((programIndex & index) == 0) {
                checkVal += src[loc++];
            }
        }
    }
    if (checkVal != num) {
        printf("Error check_packed_load_active index=%d.\n", index);
    }
}

template <typename T>
static void check_packed_load_active_neq(T *src, T *dst, const unsigned int index, unsigned int num, int count) {
    int width = ispc::width();
    int loc = 0;
    int checkVal = 0;
    for (int i = 0; i < count / width; i++) {
        for (int programIndex = 0; programIndex < width; programIndex++) {
            if ((programIndex & index) != 0) {
                checkVal += src[loc++];
            }
        }
    }
    if (checkVal != num) {
        printf("Error check_packed_load_active index=%d.\n", index);
    }
}

template <typename T>
static void check_packed_store_active_eq(T *src, T *dst, const unsigned int index, unsigned int num, int count) {
    int width = ispc::width();
    int srcLoc = 0;
    int dstLoc = 0;
    for (int i = 0; i < count / width; i++) {
        for (int programIndex = 0; programIndex < width; programIndex++) {
            if ((programIndex & index) == 0) {
                if (src[srcLoc] != dst[dstLoc++]) {
                    printf("Error check_packed_store_active loc=%d.\n", index);
                }
            }
            srcLoc++;
        }
    }
}

template <typename T>
static void check_packed_store_active_neq(T *src, T *dst, const unsigned int index, unsigned int num, int count) {
    int width = ispc::width();
    int srcLoc = 0;
    int dstLoc = 0;
    for (int i = 0; i < count / width; i++) {
        for (int programIndex = 0; programIndex < width; programIndex++) {
            if ((programIndex & index) != 0) {
                if (src[srcLoc] != dst[dstLoc++]) {
                    printf("Error check_packed_store_active loc=%d.\n", index);
                }
            }
            srcLoc++;
        }
    }
}

template <typename T>
static void check_packed_store_active2_eq(T *src, T *dst, const unsigned int index, unsigned int num, int count) {
    check_packed_store_active_eq(src, dst, index, num, count);
}

template <typename T>
static void check_packed_store_active2_neq(T *src, T *dst, const unsigned int index, unsigned int num, int count) {
    check_packed_store_active_neq(src, dst, index, num, count);
}

#define PACKED_LOAD_STORE_COND(T_C, T_ISPC, FUNC, ACTIVE_RATIO)                                                        \
    static void FUNC##_##T_ISPC##_##ACTIVE_RATIO##_eq(benchmark::State &state) {                                       \
        int count = static_cast<int>(state.range(0));                                                                  \
        T_C *dst = static_cast<T_C *>(aligned_alloc_helper(sizeof(T_C) * count));                                      \
        T_C *src = static_cast<T_C *>(aligned_alloc_helper(sizeof(T_C) * count));                                      \
        init(src, dst, count);                                                                                         \
        const unsigned int index = ACTIVE_RATIO;                                                                       \
        unsigned int num = 0;                                                                                          \
                                                                                                                       \
        for (auto _ : state) {                                                                                         \
            num = ispc::FUNC##_##T_ISPC##_eq(src, dst, index, count);                                                  \
        }                                                                                                              \
                                                                                                                       \
        check_##FUNC##_eq(src, dst, index, num, count);                                                                \
        aligned_free_helper(src);                                                                                      \
        aligned_free_helper(dst);                                                                                      \
        state.SetComplexityN(state.range(0));                                                                          \
    }                                                                                                                  \
    BENCHMARK(FUNC##_##T_ISPC##_##ACTIVE_RATIO##_eq)->ARGS;                                                            \
                                                                                                                       \
    static void FUNC##_##T_ISPC##_##ACTIVE_RATIO##_neq(benchmark::State &state) {                                      \
        int count = static_cast<int>(state.range(0));                                                                  \
        T_C *dst = static_cast<T_C *>(aligned_alloc_helper(sizeof(T_C) * count));                                      \
        T_C *src = static_cast<T_C *>(aligned_alloc_helper(sizeof(T_C) * count));                                      \
        init(src, dst, count);                                                                                         \
        const unsigned int index = ACTIVE_RATIO;                                                                       \
        unsigned int num = 0;                                                                                          \
                                                                                                                       \
        for (auto _ : state) {                                                                                         \
            num = ispc::FUNC##_##T_ISPC##_neq(src, dst, index, count);                                                 \
        }                                                                                                              \
                                                                                                                       \
        check_##FUNC##_neq(src, dst, index, num, count);                                                               \
        aligned_free_helper(src);                                                                                      \
        aligned_free_helper(dst);                                                                                      \
        state.SetComplexityN(state.range(0));                                                                          \
    }                                                                                                                  \
    BENCHMARK(FUNC##_##T_ISPC##_##ACTIVE_RATIO##_neq)->ARGS;

// The last argument constant means the following:
// 0 is all-on, all-off
// 1 is 1/2 - even, odd
// 3 is 1/4, 3/4
// 7 is 1/8, 7/8
// 15 is 1/16, 15/16

PACKED_LOAD_STORE_COND(int32_t, int32, packed_load_active, 0)
PACKED_LOAD_STORE_COND(int32_t, int32, packed_load_active, 1)
PACKED_LOAD_STORE_COND(int32_t, int32, packed_load_active, 3)
PACKED_LOAD_STORE_COND(int32_t, int32, packed_load_active, 7)
PACKED_LOAD_STORE_COND(int32_t, int32, packed_load_active, 15)

PACKED_LOAD_STORE_COND(int32_t, int32, packed_store_active, 0)
PACKED_LOAD_STORE_COND(int32_t, int32, packed_store_active, 1)
PACKED_LOAD_STORE_COND(int32_t, int32, packed_store_active, 3)
PACKED_LOAD_STORE_COND(int32_t, int32, packed_store_active, 7)
PACKED_LOAD_STORE_COND(int32_t, int32, packed_store_active, 15)

PACKED_LOAD_STORE_COND(int32_t, int32, packed_store_active2, 0)
PACKED_LOAD_STORE_COND(int32_t, int32, packed_store_active2, 1)
PACKED_LOAD_STORE_COND(int32_t, int32, packed_store_active2, 3)
PACKED_LOAD_STORE_COND(int32_t, int32, packed_store_active2, 7)
PACKED_LOAD_STORE_COND(int32_t, int32, packed_store_active2, 15)

PACKED_LOAD_STORE_COND(int64_t, int64, packed_load_active, 0)
PACKED_LOAD_STORE_COND(int64_t, int64, packed_load_active, 1)
PACKED_LOAD_STORE_COND(int64_t, int64, packed_load_active, 3)
PACKED_LOAD_STORE_COND(int64_t, int64, packed_load_active, 7)
PACKED_LOAD_STORE_COND(int64_t, int64, packed_load_active, 15)

PACKED_LOAD_STORE_COND(int64_t, int64, packed_store_active, 0)
PACKED_LOAD_STORE_COND(int64_t, int64, packed_store_active, 1)
PACKED_LOAD_STORE_COND(int64_t, int64, packed_store_active, 3)
PACKED_LOAD_STORE_COND(int64_t, int64, packed_store_active, 7)
PACKED_LOAD_STORE_COND(int64_t, int64, packed_store_active, 15)

PACKED_LOAD_STORE_COND(int64_t, int64, packed_store_active2, 0)
PACKED_LOAD_STORE_COND(int64_t, int64, packed_store_active2, 1)
PACKED_LOAD_STORE_COND(int64_t, int64, packed_store_active2, 3)
PACKED_LOAD_STORE_COND(int64_t, int64, packed_store_active2, 7)
PACKED_LOAD_STORE_COND(int64_t, int64, packed_store_active2, 15)

BENCHMARK_MAIN();
