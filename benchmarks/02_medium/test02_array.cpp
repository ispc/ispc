// Copyright (c) 2025-2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdint>
#include <stdio.h>

#include "../common.h"
#include "test02_array_ispc.h"

static Docs docs("Note:\n"
                 " - --fast-math is needed to be passed to CPP tests\n");

WARM_UP_RUN();

// Minimum size is maximum target width * 4, i.e. 64*4 = 256.
// 256 * sizeof (int) = 1kb - expected to reside in L1
// 256 * sizeof (int) << 4 = 16kb - expected to reside in L1
// 256 * sizeof (int) << 7 = 128kb - expected to reside in L2
// 256 * sizeof (int) << 12 = 4 Mb - expected to reside in L3.
#define ARGS Arg(256)->Arg(256 << 4)->Arg(256 << 7)->Arg(256 << 12)
// #define ARGS Arg(100)->Arg(1000)->Arg(10000)

// Relative tolerance. The float sum of the largest input (1M elements) differs from the exact one by up to
// about 5e-4 depending on the summation order, and the error grows with the size.
const double eps = 1e-3;

static void init_linear(float *src, int count) {
    for (int i = 0; i < count; i++) {
        src[i] = static_cast<float>(i) * 0.001f;
    }
}

void array_mean_cpp(float *src, float *dst, int count) {
    float sum = 0;
    for (int i = 0; i < count; i++) {
        sum += src[i];
    }
    *dst = sum / count;
}

static bool check(float *src, float dst, int count) {
    double sum = 0;
    for (int i = 0; i < count; i++) {
        sum += src[i];
    }
    double expected = sum / count;
    if (std::abs(expected - dst) > eps * std::abs(expected)) {
        printf("Error expected %g, return %g\n", expected, dst);
        return false;
    }
    return true;
}

static void ArrayMeanISPC(benchmark::State &state) {
    int count = static_cast<int>(state.range(0));
    float *src = static_cast<float *>(aligned_alloc_helper(sizeof(float) * count));
    float dst = 0;
    init_linear(src, count);

    for (auto _ : state) {
        ispc::ArrayMean(src, dst, count);
    }

    if (!check(src, dst, count))
        fail_benchmark(state);
    aligned_free_helper(src);
}
BENCHMARK(ArrayMeanISPC)->ARGS;

static void ArrayMeanCPP(benchmark::State &state) {
    int count = static_cast<int>(state.range(0));
    float *src = static_cast<float *>(aligned_alloc_helper(sizeof(float) * count));
    float dst = 0;
    init_linear(src, count);

    for (auto _ : state) {
        array_mean_cpp(src, &dst, count);
    }

    if (!check(src, dst, count))
        fail_benchmark(state);
    aligned_free_helper(src);
}
BENCHMARK(ArrayMeanCPP)->ARGS;

CHECKED_BENCHMARK_MAIN();
