#include <benchmark/benchmark.h>
#include <iostream>

#include "../common.h"

#include "07_loop_unroll_varying_ispc.h"

static Docs docs("Check the performance and functionality of loop unrolling with varying types.\n"
                 "The bench kernel performs prefix scan over an input buffer.\n"
                 "[int8, uint8, int16, uint16, int32, uint32, int64, uint64, float, double]"
                 " x [for, foreach] x [ nounroll, unroll(2), unroll(4) ] versions.\n");

WARM_UP_RUN();

template <typename T> void init_src(T *buf, size_t num_elems) {
    for (int i = 0; i < static_cast<int>(num_elems); i++) {
        const T elem = static_cast<T>(i);
        buf[i] = elem * elem;
    }
}

template <typename T> void init_dst(T *buf, size_t num_elems) {
    for (int i = 0; i < static_cast<int>(num_elems); i++) {
        buf[i] = static_cast<T>(0);
    }
}

template <typename T> void check(T *src, T *dst, size_t num_elems) {
    // Allocate a buffer to perform the computation (prefix scan)
    T *buf = static_cast<T *>(aligned_alloc_helper(sizeof(T) * num_elems));
    if (buf == nullptr) {
        std::cerr << "[07_loop_unroll_varying] Failed to allocate buffer." << std::endl;
        return;
    }

    // Perform the prefix scan
    buf[0] = src[0];
    for (int i = 1; i < static_cast<int>(num_elems); i++) {
        buf[i] = src[i] + src[i - 1];
    }

    // Check the output with the previously computed destination buffer
    for (int i = 0; i < static_cast<int>(num_elems); i++) {
        if (buf[i] != dst[i]) {
            std::cerr << "[07_loop_unroll_varying] Error: "
                      << "buf[" << i << "] != dst[" << i << "]"
                      << "(" << buf[i] << " != " << dst[i] << ")" << std::endl;
            break;
        }
    }

    aligned_free_helper(buf);
}

// Minimum size is maximum target width * 4, i.e. 64*4 = 256.
// // 256 * sizeof (int) = 1kb - expected to reside in L1
// // 256 * sizeof (int) << 4 = 16kb - expected to reside in L1
// // 256 * sizeof (int) << 7 = 128kb - expected to reside in L2
// // 256 * sizeof (int) << 12 = 4 Mb - expected to reside in L3.
#define ARGS Arg(256)->Arg(256 << 4)->Arg(256 << 7)->Arg(256 << 12)

#ifndef UNROLL_VARYING_BENCH
#define UNROLL_VARYING_BENCH(T_FOR, T_C, T_ISPC, UNROLL_FACTOR)                                                        \
    static void unroll_varying_##T_FOR##_##T_ISPC##_##UNROLL_FACTOR(benchmark::State &state) {                         \
        const size_t num_elems = state.range(0);                                                                       \
        T_C *src = static_cast<T_C *>(aligned_alloc_helper(sizeof(T_C) * num_elems));                                  \
        T_C *dst = static_cast<T_C *>(aligned_alloc_helper(sizeof(T_C) * num_elems));                                  \
        init_src(src, num_elems);                                                                                      \
        init_dst(dst, num_elems);                                                                                      \
        for (auto _ : state) {                                                                                         \
            ispc::unroll_varying_##T_FOR##_##T_ISPC##_##UNROLL_FACTOR(src, dst, num_elems);                            \
        }                                                                                                              \
        check(src, dst, num_elems);                                                                                    \
        aligned_free_helper(src);                                                                                      \
        aligned_free_helper(dst);                                                                                      \
        state.SetComplexityN(state.range(0));                                                                          \
    }                                                                                                                  \
    BENCHMARK(unroll_varying_##T_FOR##_##T_ISPC##_##UNROLL_FACTOR)->ARGS
#endif // UNROLL_VARYING_BENCH

UNROLL_VARYING_BENCH(foreach, uint64_t, uint64, 1);
UNROLL_VARYING_BENCH(foreach, int64_t, int64, 1);
UNROLL_VARYING_BENCH(foreach, uint32_t, uint32, 1);
UNROLL_VARYING_BENCH(foreach, int32_t, int32, 1);
UNROLL_VARYING_BENCH(foreach, uint16_t, uint16, 1);
UNROLL_VARYING_BENCH(foreach, int16_t, int16, 1);
UNROLL_VARYING_BENCH(foreach, uint8_t, uint8, 1);
UNROLL_VARYING_BENCH(foreach, int8_t, int8, 1);
UNROLL_VARYING_BENCH(foreach, double, double, 1);
UNROLL_VARYING_BENCH(foreach, float, float, 1);

UNROLL_VARYING_BENCH(foreach, uint64_t, uint64, 2);
UNROLL_VARYING_BENCH(foreach, int64_t, int64, 2);
UNROLL_VARYING_BENCH(foreach, uint32_t, uint32, 2);
UNROLL_VARYING_BENCH(foreach, int32_t, int32, 2);
UNROLL_VARYING_BENCH(foreach, uint16_t, uint16, 2);
UNROLL_VARYING_BENCH(foreach, int16_t, int16, 2);
UNROLL_VARYING_BENCH(foreach, uint8_t, uint8, 2);
UNROLL_VARYING_BENCH(foreach, int8_t, int8, 2);
UNROLL_VARYING_BENCH(foreach, double, double, 2);
UNROLL_VARYING_BENCH(foreach, float, float, 2);

UNROLL_VARYING_BENCH(foreach, uint64_t, uint64, 4);
UNROLL_VARYING_BENCH(foreach, int64_t, int64, 4);
UNROLL_VARYING_BENCH(foreach, uint32_t, uint32, 4);
UNROLL_VARYING_BENCH(foreach, int32_t, int32, 4);
UNROLL_VARYING_BENCH(foreach, uint16_t, uint16, 4);
UNROLL_VARYING_BENCH(foreach, int16_t, int16, 4);
UNROLL_VARYING_BENCH(foreach, uint8_t, uint8, 4);
UNROLL_VARYING_BENCH(foreach, int8_t, int8, 4);
UNROLL_VARYING_BENCH(foreach, double, double, 4);
UNROLL_VARYING_BENCH(foreach, float, float, 4);

UNROLL_VARYING_BENCH(for,     uint64_t,  uint64, 1);
UNROLL_VARYING_BENCH(for,     int64_t,   int64,  1);
UNROLL_VARYING_BENCH(for,     uint32_t,  uint32, 1);
UNROLL_VARYING_BENCH(for,     int32_t,   int32,  1);
UNROLL_VARYING_BENCH(for,     uint16_t,  uint16, 1);
UNROLL_VARYING_BENCH(for,     int16_t,   int16,  1);
UNROLL_VARYING_BENCH(for,     uint8_t,   uint8,  1);
UNROLL_VARYING_BENCH(for,     int8_t,    int8,   1);
UNROLL_VARYING_BENCH(for,     double,    double, 1);
UNROLL_VARYING_BENCH(for,     float,     float,  1);

UNROLL_VARYING_BENCH(for,     uint64_t,  uint64, 2);
UNROLL_VARYING_BENCH(for,     int64_t,   int64,  2);
UNROLL_VARYING_BENCH(for,     uint32_t,  uint32, 2);
UNROLL_VARYING_BENCH(for,     int32_t,   int32,  2);
UNROLL_VARYING_BENCH(for,     uint16_t,  uint16, 2);
UNROLL_VARYING_BENCH(for,     int16_t,   int16,  2);
UNROLL_VARYING_BENCH(for,     uint8_t,   uint8,  2);
UNROLL_VARYING_BENCH(for,     int8_t,    int8,   2);
UNROLL_VARYING_BENCH(for,     double,    double, 2);
UNROLL_VARYING_BENCH(for,     float,     float,  2);

UNROLL_VARYING_BENCH(for,     uint64_t,  uint64, 4);
UNROLL_VARYING_BENCH(for,     int64_t,   int64,  4);
UNROLL_VARYING_BENCH(for,     uint32_t,  uint32, 4);
UNROLL_VARYING_BENCH(for,     int32_t,   int32,  4);
UNROLL_VARYING_BENCH(for,     uint16_t,  uint16, 4);
UNROLL_VARYING_BENCH(for,     int16_t,   int16,  4);
UNROLL_VARYING_BENCH(for,     uint8_t,   uint8,  4);
UNROLL_VARYING_BENCH(for,     int8_t,    int8,   4);
UNROLL_VARYING_BENCH(for,     double,    double, 4);
UNROLL_VARYING_BENCH(for,     float,     float,  4);

BENCHMARK_MAIN();
