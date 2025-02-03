// Copyright (c) 2025, Intel Corporation
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

const float eps = 0.00001f;

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

template <typename F> static void check(float *src, float dst, int count, F fp) {
    float expected = 0;
    fp(src, &expected, count);
    if (std::abs(expected - dst) > eps) {
        printf("Error expected %g, return %g\n", expected, dst);
        return;
    }
}

static void ArrayMeanISPC(benchmark::State &state) {
    int count = static_cast<int>(state.range(0));
    // float *src = static_cast<float *>(aligned_alloc_helper(sizeof(float) * count));
    float *src = new float[count];
    float dst = 0;
    init_linear(src, count);

    for (auto _ : state) {
        ispc::ArrayMean(src, dst, count);
    }

    check(src, dst, count, array_mean_cpp);
    // aligned_free_helper(src);
    delete[] src;
}
BENCHMARK(ArrayMeanISPC)->ARGS;

static void ArrayMeanCPP(benchmark::State &state) {
    int count = static_cast<int>(state.range(0));
    // float *src = static_cast<float *>(aligned_alloc_helper(sizeof(float) * count));
    float *src = new float[count];
    float dst = 0;
    init_linear(src, count);

    for (auto _ : state) {
        array_mean_cpp(src, &dst, count);
    }

    check(src, dst, count, array_mean_cpp);
    // aligned_free_helper(src);
    delete[] src;
}
BENCHMARK(ArrayMeanCPP)->ARGS;

BENCHMARK_MAIN();
