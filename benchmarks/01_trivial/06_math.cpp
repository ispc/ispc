#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdint>
#include <stdio.h>

#include "../common.h"
#include "06_math_ispc.h"

static Docs docs("Check perfomance of math functions.\n"
                 "Things to note:\n"
                 " - benchmarks are focused on performance, not accuracy verification.\n"
                 " - math functions are invoked in unmasked context.\n"
                 " - workload sizes are designed to hit different caches (L1/L2/L3).\n"
                 "Expectations:\n"
                 " - No regressions\n");

// Minimum size is maximum target width * 4, i.e. 64*4 = 256.
// 256 * sizeof (int) = 1kb - expected to reside in L1
// 256 * sizeof (int) << 4 = 16kb - expected to reside in L1
// 256 * sizeof (int) << 7 = 128kb - expected to reside in L2
// 256 * sizeof (int) << 12 = 4 Mb - expected to reside in L3.
#define ARGS Arg(256)->Arg(256 << 4)->Arg(256 << 7)->Arg(256 << 12)

// Helper functions
const double PI = 3.141592653589793;

template <typename T> static void init_linear(T *src, T *dst, int count) {
    for (int i = 0; i < count; i++) {
        src[i] = static_cast<T>(i + 1);
        dst[i] = 0;
    }
}

// First argument is linear in range [1, 31], second argument is in range of [0, 2].
template <typename T> static void init_linear2(T *src1, T *src2, T *dst, int count) {
    for (int i = 0; i < count; i++) {
        src1[i] = static_cast<T>((i % 30) + 1);
        src2[i] = static_cast<T>(std::fmod(static_cast<T>(i) * PI, 2.0));
        dst[i] = 0;
    }
}

// First argument is linear, second argument is linear in range of [0, 20].
template <typename T> static void init_ldexp(T *src1, int *src2, T *dst, int count) {
    for (int i = 0; i < count; i++) {
        src1[i] = static_cast<T>(i + 1);
        src2[i] = i % 20;
        dst[i] = 0;
    }
}

// The source argument is linear, two destinations.
template <typename T> static void init_frexp(T *src, T *dst1, int *dst2, int count) {
    for (int i = 0; i < count; i++) {
        src[i] = static_cast<T>(i + 1);
        dst1[i] = 0;
        dst2[i] = 0;
    }
}

// Generate numbers in the range (-pi, pi).
template <typename T> static void init_pi(T *src, T *dst, int count) {
    for (int i = 0; i < count; i++) {
        src[i] = static_cast<T>(std::fmod(static_cast<T>(i), PI) - (PI / 2));
        dst[i] = 0;
    }
}

// Generate numbers in the range (-pi/2+eps, pi/2-eps).
// Use eps, to avoid precision problems for tan().
template <typename T> static void init_half_pi(T *src, T *dst, int count) {
    T eps = 0.01f;
    for (int i = 0; i < count; i++) {
        src[i] = static_cast<T>(std::fmod(static_cast<T>(i), (PI - 2 * eps)) - (PI / 2 - eps));
        dst[i] = 0;
    }
}

// Generate numbers in the range (-1+eps, 1-eps).
// Use eps, to avoid precision problems for atan().
template <typename T> static void init_one(T *src, T *dst, int count) {
    T eps = 0.01f;
    for (int i = 0; i < count; i++) {
        src[i] = static_cast<T>(std::fmod(static_cast<T>(i) * PI, 2.0 - 2 * eps) - (1.0 - eps));
        dst[i] = 0;
    }
}

template <typename T> static void check_sqrt(T *src, T *dst, int count) {
    T eps = 0.001f;
    for (int i = 0; i < count; i++) {
        T expected = std::sqrt(src[i]);
        if (std::abs(expected - dst[i]) > eps) {
            printf("Error i=%d, expected %g, return %g\n", i, expected, dst[i]);
            return;
        }
    }
}

template <typename T> static void check_rsqrt(T *src, T *dst, int count) {
    T eps = 0.001f;
    for (int i = 0; i < count; i++) {
        T expected = 1.0 / std::sqrt(src[i]);
        if (std::abs(expected - dst[i]) > eps) {
            printf("Error i=%d, expected %g, return %g\n", i, expected, dst[i]);
            return;
        }
    }
}

template <typename T> static void check_rsqrt_fast(T *src, T *dst, int count) { check_rsqrt(src, dst, count); }

template <typename T> static void check_rcp(T *src, T *dst, int count) {
    T eps = 0.001f;
    for (int i = 0; i < count; i++) {
        T expected = 1.0 / src[i];
        if (std::abs(expected - dst[i]) > eps) {
            printf("Error i=%d, expected %g, return %g\n", i, expected, dst[i]);
            return;
        }
    }
}

template <typename T> static void check_rcp_fast(T *src, T *dst, int count) { check_rcp(src, dst, count); }

template <typename T> static void check_ldexp(T *src1, int *src2, T *dst, int count) {
    T eps = 0.001f;
    for (int i = 0; i < count; i++) {
        T expected = std::ldexp(src1[i], src2[i]);
        if (std::abs(expected - dst[i]) > eps) {
            printf("Error i=%d, expected %g, return %g\n", i, expected, dst[i]);
            return;
        }
    }
}

template <typename T> static void check_frexp(T *src, T *dst1, int *dst2, int count) {
    T eps = 0.001f;
    for (int i = 0; i < count; i++) {
        int pow = 0;
        T expected = std::frexp(src[i], &pow);
        if (std::abs(expected - dst1[i]) > eps || pow != dst2[i]) {
            printf("Error i=%d, expected %g, return %g\n", i, expected, dst1[i]);
            return;
        }
    }
}

template <typename T> static void check_sin(T *src, T *dst, int count) {
    T eps = 0.001f;
    for (int i = 0; i < count; i++) {
        T expected = std::sin(src[i]);
        if (std::abs(expected - dst[i]) > eps) {
            printf("Error i=%d, expected %g, return %g\n", i, expected, dst[i]);
            return;
        }
    }
}

template <typename T> static void check_asin(T *src, T *dst, int count) {
    T eps = 0.001f;
    for (int i = 0; i < count; i++) {
        T expected = std::asin(src[i]);
        if (std::abs(expected - dst[i]) > eps) {
            printf("Error i=%d, expected %g, return %g\n", i, expected, dst[i]);
            return;
        }
    }
}

template <typename T> static void check_cos(T *src, T *dst, int count) {
    T eps = 0.001f;
    for (int i = 0; i < count; i++) {
        T expected = std::cos(src[i]);
        if (std::abs(expected - dst[i]) > eps) {
            printf("Error i=%d, expected %g, return %g\n", i, expected, dst[i]);
            return;
        }
    }
}

template <typename T> static void check_acos(T *src, T *dst, int count) {
    T eps = 0.001f;
    for (int i = 0; i < count; i++) {
        T expected = std::acos(src[i]);
        if (std::abs(expected - dst[i]) > eps) {
            printf("Error i=%d, expected %g, return %g\n", i, expected, dst[i]);
            return;
        }
    }
}

template <typename T> static void check_tan(T *src, T *dst, int count) {
    T eps = 0.001f;
    for (int i = 0; i < count; i++) {
        T expected = std::tan(src[i]);
        if (std::abs(expected - dst[i]) > eps) {
            printf("Error i=%d, expected %g, return %g\n", i, expected, dst[i]);
            return;
        }
    }
}

template <typename T> static void check_atan(T *src, T *dst, int count) {
    T eps = 0.001f;
    for (int i = 0; i < count; i++) {
        T expected = std::atan(src[i]);
        if (std::abs(expected - dst[i]) > eps) {
            printf("Error i=%d, expected %g, return %g\n", i, expected, dst[i]);
            return;
        }
    }
}

template <typename T> static void check_atan2(T *src1, T *src2, T *dst, int count) {
    T eps = 0.001f;
    for (int i = 0; i < count; i++) {
        T expected = std::atan2(src1[i], src2[i]);
        if (std::abs(expected - dst[i]) > eps) {
            printf("Error i=%d, expected %g, return %g\n", i, expected, dst[i]);
            return;
        }
    }
}

template <typename T> static void check_exp(T *src, T *dst, int count) {
    T eps = 0.001f;
    for (int i = 0; i < count; i++) {
        T expected = std::exp(src[i]);
        if (std::abs(expected - dst[i]) > eps) {
            printf("Error i=%d, expected %g, return %g\n", i, expected, dst[i]);
            return;
        }
    }
}

template <typename T> static void check_log(T *src, T *dst, int count) {
    T eps = 0.001f;
    for (int i = 0; i < count; i++) {
        T expected = std::log(src[i]);
        if (std::abs(expected - dst[i]) > eps) {
            printf("Error i=%d, expected %g, return %g\n", i, expected, dst[i]);
            return;
        }
    }
}

template <typename T> static void check_pow(T *src1, T *src2, T *dst, int count) {
    T eps = 0.001f;
    for (int i = 0; i < count; i++) {
        T expected = std::pow(src1[i], src2[i]);
        if (std::abs(expected - dst[i]) > eps) {
            printf("Error i=%d, expected %g, return %g\n", i, expected, dst[i]);
            return;
        }
    }
}

// Functions with single source, single destination.
#define TEST(NAME, T, INIT)                                                                                            \
    static void NAME##_##T(benchmark::State &state) {                                                                  \
        int count = static_cast<int>(state.range(0));                                                                  \
        T *src = static_cast<T *>(aligned_alloc_helper(sizeof(T) * count));                                            \
        T *dst = static_cast<T *>(aligned_alloc_helper(sizeof(T) * count));                                            \
        INIT(src, dst, count);                                                                                         \
                                                                                                                       \
        for (auto _ : state) {                                                                                         \
            ispc::NAME##_##T(src, dst, count);                                                                         \
        }                                                                                                              \
                                                                                                                       \
        check_##NAME(src, dst, count);                                                                                 \
        aligned_free_helper(src);                                                                                      \
        aligned_free_helper(dst);                                                                                      \
    }                                                                                                                  \
    BENCHMARK(NAME##_##T)->ARGS;

// Functions with two sources of the same type, single destination.
#define TEST2(NAME, T, INIT)                                                                                           \
    static void NAME##_##T(benchmark::State &state) {                                                                  \
        int count = static_cast<int>(state.range(0));                                                                  \
        T *src1 = static_cast<T *>(aligned_alloc_helper(sizeof(T) * count));                                           \
        T *src2 = static_cast<T *>(aligned_alloc_helper(sizeof(T) * count));                                           \
        T *dst = static_cast<T *>(aligned_alloc_helper(sizeof(T) * count));                                            \
        INIT(src1, src2, dst, count);                                                                                  \
                                                                                                                       \
        for (auto _ : state) {                                                                                         \
            ispc::NAME##_##T(src1, src2, dst, count);                                                                  \
        }                                                                                                              \
                                                                                                                       \
        check_##NAME(src1, src2, dst, count);                                                                          \
        aligned_free_helper(src1);                                                                                     \
        aligned_free_helper(src2);                                                                                     \
        aligned_free_helper(dst);                                                                                      \
    }                                                                                                                  \
    BENCHMARK(NAME##_##T)->ARGS;

// Functions with two sources of type T and int, single destination.
#define TEST3(NAME, T, INIT)                                                                                           \
    static void NAME##_##T(benchmark::State &state) {                                                                  \
        int count = static_cast<int>(state.range(0));                                                                  \
        T *src1 = static_cast<T *>(aligned_alloc_helper(sizeof(T) * count));                                           \
        int *src2 = static_cast<int *>(aligned_alloc_helper(sizeof(int) * count));                                     \
        T *dst = static_cast<T *>(aligned_alloc_helper(sizeof(T) * count));                                            \
        INIT(src1, src2, dst, count);                                                                                  \
                                                                                                                       \
        for (auto _ : state) {                                                                                         \
            ispc::NAME##_##T(src1, src2, dst, count);                                                                  \
        }                                                                                                              \
                                                                                                                       \
        check_##NAME(src1, src2, dst, count);                                                                          \
        aligned_free_helper(src1);                                                                                     \
        aligned_free_helper(src2);                                                                                     \
        aligned_free_helper(dst);                                                                                      \
    }                                                                                                                  \
    BENCHMARK(NAME##_##T)->ARGS;

// Functions with single source and two destinations (T and int).
#define TEST4(NAME, T, INIT)                                                                                           \
    static void NAME##_##T(benchmark::State &state) {                                                                  \
        int count = static_cast<int>(state.range(0));                                                                  \
        T *src = static_cast<T *>(aligned_alloc_helper(sizeof(T) * count));                                            \
        T *dst1 = static_cast<T *>(aligned_alloc_helper(sizeof(T) * count));                                           \
        int *dst2 = static_cast<int *>(aligned_alloc_helper(sizeof(int) * count));                                     \
        INIT(src, dst1, dst2, count);                                                                                  \
                                                                                                                       \
        for (auto _ : state) {                                                                                         \
            ispc::NAME##_##T(src, dst1, dst2, count);                                                                  \
        }                                                                                                              \
                                                                                                                       \
        check_##NAME(src, dst1, dst2, count);                                                                          \
        aligned_free_helper(src);                                                                                      \
        aligned_free_helper(dst1);                                                                                     \
        aligned_free_helper(dst2);                                                                                     \
    }                                                                                                                  \
    BENCHMARK(NAME##_##T)->ARGS;

TEST(sqrt, float, init_linear)
TEST(sqrt, double, init_linear)
TEST(rsqrt, float, init_linear)
TEST(rsqrt, double, init_linear)
TEST(rsqrt_fast, float, init_linear)
TEST(rsqrt_fast, double, init_linear)
TEST(rcp, float, init_linear)
TEST(rcp, double, init_linear)
TEST(rcp_fast, float, init_linear)
TEST(rcp_fast, double, init_linear)
TEST3(ldexp, float, init_ldexp)
TEST3(ldexp, double, init_ldexp)
TEST4(frexp, float, init_frexp)
TEST4(frexp, double, init_frexp)

TEST(sin, float, init_pi)
TEST(sin, double, init_pi)
TEST(asin, float, init_one)
TEST(asin, double, init_one)
TEST(cos, float, init_pi)
TEST(cos, double, init_pi)
TEST(acos, float, init_one)
TEST(acos, double, init_one)
TEST(tan, float, init_half_pi)
TEST(tan, double, init_half_pi)
TEST(atan, float, init_linear)
TEST(atan, double, init_linear)
TEST2(atan2, float, init_linear2)
TEST2(atan2, double, init_linear2)

TEST(exp, float, init_pi)
TEST(exp, double, init_pi)
TEST(log, float, init_linear)
TEST(log, double, init_linear)
TEST2(pow, float, init_linear2)
TEST2(pow, double, init_linear2)

BENCHMARK_MAIN();
