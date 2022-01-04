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

WARM_UP_RUN();

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

template <typename T, typename F> static void check(T *src, T *dst, int count, F fp) {
    T eps = 0.001f;
    for (int i = 0; i < count; i++) {
        T expected = fp(src[i]);
        if (std::abs(expected - dst[i]) > eps) {
            printf("Error i=%d, expected %g, return %g\n", i, expected, dst[i]);
            return;
        }
    }
}

template <typename T, typename F> static void check2(T *src1, T *src2, T *dst, int count, F fp) {
    T eps = 0.001f;
    for (int i = 0; i < count; i++) {
        T expected = fp(src1[i], src2[i]);
        if (std::abs(expected - dst[i]) > eps) {
            printf("Error i=%d, expected %g, return %g\n", i, expected, dst[i]);
            return;
        }
    }
}

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

// Functions with single source, single destination.
#define TEST(NAME, T, INIT, CHECK)                                                                                     \
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
        check(src, dst, count, [](T x) { return CHECK; });                                                             \
        aligned_free_helper(src);                                                                                      \
        aligned_free_helper(dst);                                                                                      \
    }                                                                                                                  \
    BENCHMARK(NAME##_##T)->ARGS;

// Functions with two sources of the same type, single destination.
#define TEST2(NAME, T, INIT, CHECK)                                                                                    \
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
        check2(src1, src2, dst, count, [](T x, T y) { return CHECK; });                                                \
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

TEST(sqrt, float, init_linear, std::sqrt(x))
TEST(sqrt, double, init_linear, std::sqrt(x))
TEST(rsqrt, float, init_linear, 1.0f / std::sqrt(x))
TEST(rsqrt, double, init_linear, 1.0 / std::sqrt(x))
TEST(rsqrt_fast, float, init_linear, 1.0f / std::sqrt(x))
TEST(rsqrt_fast, double, init_linear, 1.0 / std::sqrt(x))
TEST(rcp, float, init_linear, 1.0f / x)
TEST(rcp, double, init_linear, 1.0 / x)
TEST(rcp_fast, float, init_linear, 1.0f / x)
TEST(rcp_fast, double, init_linear, 1.0 / x)
TEST3(ldexp, float, init_ldexp)
TEST3(ldexp, double, init_ldexp)
TEST4(frexp, float, init_frexp)
TEST4(frexp, double, init_frexp)

TEST(sin, float, init_pi, std::sin(x))
TEST(sin, double, init_pi, std::sin(x))
TEST(asin, float, init_one, std::asin(x))
TEST(asin, double, init_one, std::asin(x))
TEST(cos, float, init_pi, std::cos(x))
TEST(cos, double, init_pi, std::cos(x))
TEST(acos, float, init_one, std::acos(x))
TEST(acos, double, init_one, std::acos(x))
TEST(tan, float, init_half_pi, std::tan(x))
TEST(tan, double, init_half_pi, std::tan(x))
TEST(atan, float, init_linear, std::atan(x))
TEST(atan, double, init_linear, std::atan(x))
TEST2(atan2, float, init_linear2, std::atan2(x, y))
TEST2(atan2, double, init_linear2, std::atan2(x, y))

TEST(exp, float, init_pi, std::exp(x))
TEST(exp, double, init_pi, std::exp(x))
TEST(log, float, init_linear, std::log(x))
TEST(log, double, init_linear, std::log(x))
TEST2(pow, float, init_linear2, std::pow(x, y))
TEST2(pow, double, init_linear2, std::pow(x, y))

BENCHMARK_MAIN();
