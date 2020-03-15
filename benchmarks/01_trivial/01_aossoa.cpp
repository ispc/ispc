#include <benchmark/benchmark.h>
#include <cstdint>
#include <stdio.h>

#include "../common.h"
#include "01_aossoa_ispc.h"

static Docs docs("aos_to_soa*_stdlib_<type> - test for stdlib implimentation for different types\n"
                 "aos_to_soa*_ispc_<type> - test for ISPC implementation of these library functions.\n"
                 "Expectations:\n"
                 " - stdlib functions have the same speed for float vs int32 and double vs int64.\n"
                 " - stdlib implementation is faster or has same performance as ISPC implementation.\n");

// Minimum size is maximum target width * 4, i.e. 64*4 = 256.
// 256 * sizeof (int) = 1kb - expected to reside in L1
// 256 * sizeof (int) << 4 = 16kb - expected to reside in L1
// 256 * sizeof (int) << 7 = 128kb - expected to reside in L2
// 256 * sizeof (int) << 12 = 4 Mb - expected to reside in L3.
#define ARGS4 Arg(256)->Arg(256 << 4)->Arg(256 << 7)->Arg(256 << 12)
#define ARGS3 Arg(192)->Arg(192 << 4)->Arg(192 << 7)->Arg(192 << 12)

// Helper functions
template <typename T> static void init(T *src, T *dst, int count) {
    for (int i = 0; i < count; i++) {
        src[i] = static_cast<T>(i);
        dst[i] = 0;
    }
}

template <typename T> static void check4(T *dst, int count) {
    int width = ispc::width();
    int chunk = width * 4;
    for (int i = 0; i < count; i++) {
        int base = (i / chunk) * chunk;
        int in_chunk = i % chunk;
        int vec_num = in_chunk / width;
        int in_vec = in_chunk % width;
        int expected = base + in_vec * 4 + vec_num;
        // Note, we use == comparison even for floating point types, as we expect that
        // values are only copied, but not manipulated in any other way.
        if (dst[i] != static_cast<T>(expected)) {
            printf("Error i=%d, expected %d, returned %d\n", i, expected, static_cast<int>(dst[i]));
            return;
        }
    }
}

template <typename T> static void check3(T *dst, int count) {
    int width = ispc::width();
    int chunk = width * 3;
    for (int i = 0; i < count; i++) {
        int base = (i / chunk) * chunk;
        int in_chunk = i % chunk;
        int vec_num = in_chunk / width;
        int in_vec = in_chunk % width;
        int expected = base + in_vec * 3 + vec_num;
        // Note, we use == comparison even for floating point types, as we expect that
        // values are only copied, but not manipulated in any other way.
        if (dst[i] != static_cast<T>(expected)) {
            printf("Error i=%d, expected %d, returned %d\n", i, expected, static_cast<int>(dst[i]));
            return;
        }
    }
}

#define AOS_TO_SOA_STDLIB(N, T_C, T_ISPC)                                                                              \
    static void aos_to_soa##N##_stdlib_##T_ISPC(benchmark::State &state) {                                             \
        int count = state.range(0);                                                                                    \
        T_C *src = static_cast<T_C *>(aligned_alloc(ALIGNMENT, sizeof(T_C) * count));                                  \
        T_C *dst = static_cast<T_C *>(aligned_alloc(ALIGNMENT, sizeof(T_C) * count));                                  \
        init(src, dst, count);                                                                                         \
                                                                                                                       \
        for (auto _ : state) {                                                                                         \
            ispc::aos_to_soa##N##_stdlib_##T_ISPC(src, dst, count);                                                    \
        }                                                                                                              \
                                                                                                                       \
        check##N(dst, count);                                                                                          \
        free(src);                                                                                                     \
        free(dst);                                                                                                     \
    }                                                                                                                  \
    BENCHMARK(aos_to_soa##N##_stdlib_##T_ISPC)->ARGS##N;

#define AOS_TO_SOA_ISPC(N, T_C, T_ISPC)                                                                                \
    static void aos_to_soa##N##_ispc_##T_ISPC(benchmark::State &state) {                                               \
        int count = state.range(0);                                                                                    \
        T_C *src = static_cast<T_C *>(aligned_alloc(ALIGNMENT, sizeof(T_C) * count));                                  \
        T_C *dst = static_cast<T_C *>(aligned_alloc(ALIGNMENT, sizeof(T_C) * count));                                  \
        init(src, dst, count);                                                                                         \
                                                                                                                       \
        for (auto _ : state) {                                                                                         \
            ispc::aos_to_soa##N##_ispc_##T_ISPC(src, dst, count);                                                      \
        }                                                                                                              \
                                                                                                                       \
        check##N(dst, count);                                                                                          \
        free(src);                                                                                                     \
        free(dst);                                                                                                     \
    }                                                                                                                  \
    BENCHMARK(aos_to_soa##N##_ispc_##T_ISPC)->ARGS##N;

AOS_TO_SOA_STDLIB(4, int, int32);
AOS_TO_SOA_STDLIB(4, float, float);
AOS_TO_SOA_STDLIB(4, int64_t, int64);
AOS_TO_SOA_STDLIB(4, double, double);

AOS_TO_SOA_ISPC(4, int, int32);
AOS_TO_SOA_ISPC(4, float, float);
AOS_TO_SOA_ISPC(4, int64_t, int64);
AOS_TO_SOA_ISPC(4, double, double);

AOS_TO_SOA_STDLIB(3, int, int32);
AOS_TO_SOA_STDLIB(3, float, float);
AOS_TO_SOA_STDLIB(3, int64_t, int64);
AOS_TO_SOA_STDLIB(3, double, double);

AOS_TO_SOA_ISPC(3, int, int32);
AOS_TO_SOA_ISPC(3, float, float);
AOS_TO_SOA_ISPC(3, int64_t, int64);
AOS_TO_SOA_ISPC(3, double, double);

BENCHMARK_MAIN();
