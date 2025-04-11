// Copyright (c) 2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdint>
#include <stdio.h>

#include "../common.h"
#include "test03_int_div_ispc.h"

static Docs docs("Note:\n"
                 " - --fast-math is needed to be passed to CPP tests\n");

WARM_UP_RUN();

// Minimum size is maximum target width * 4, i.e. 64*4 = 256.
// 256 * sizeof (int) = 1kb - expected to reside in L1
// 256 * sizeof (int) << 4 = 16kb - expected to reside in L1
// 256 * sizeof (int) << 7 = 128kb - expected to reside in L2
// 256 * sizeof (int) << 12 = 4 Mb - expected to reside in L3.
#define ARGS Arg(256)->Arg(256 << 4)->Arg(256 << 7)->Arg(256 << 12)

static void init_random(int *src, int count) {
    for (int i = 0; i < count; i++) {
        src[i] = 1 + rand() % 100000;
    }
}

static void init_full_mask(int *mask, int count) {
    for (int i = 0; i < count; i++) {
        mask[i] = 1;
    }
}

static void init_half_mask(int *mask, int count) {
    for (int i = 0; i < count; i++) {
        mask[i] = (rand() % 2 == 0) ? 1 : 0;
    }
}

static void init_quat_mask(int *mask, int count) {
    for (int i = 0; i < count; i++) {
        mask[i] = (rand() % 4 == 0) ? 1 : 0;
    }
}

int div_i(int *src0, int *src1, int *mask, int i) { return src0[i] / src1[i]; }

int div_i_mask(int *src0, int *src1, int *mask, int i) { return (mask[i] == 1) ? (src0[i] / src1[i]) : 0; }

void array_int_div_full_cpp(int *src0, int *src1, int *dst, int *mask, int count) {
    for (int i = 0; i < count; i++) {
        dst[i] = div_i(src0, src1, mask, i);
    }
}

void array_int_div_mask_cpp(int *src0, int *src1, int *dst, int *mask, int count) {
    for (int i = 0; i < count; i++) {
        dst[i] = div_i_mask(src0, src1, mask, i);
    }
}

template <typename F> static void check(int *src0, int *src1, int *dst, int *mask, int count, F check_foo) {
    for (int i = 0; i < count; i++) {
        int ans = check_foo(src0, src1, mask, i);
        if (dst[i] != ans) {
            printf("Error at %d: expected %d, return %d\n", i, ans, dst[i]);
            return;
        }
    }
}

static void ArrayIntDivFullISPC(benchmark::State &state) {
    int count = static_cast<int>(state.range(0));
    int *src0 = new int[count];
    int *src1 = new int[count];
    int *dst = new int[count];
    int *mask = new int[count];
    init_random(src0, count);
    init_random(src1, count);
    init_full_mask(mask, count);

    for (auto _ : state) {
        ispc::ArrayIntDivFull(src0, src1, dst, mask, count);
    }

    check(src0, src1, dst, mask, count, div_i);
    delete[] src0;
    delete[] src1;
    delete[] dst;
    delete[] mask;
}
BENCHMARK(ArrayIntDivFullISPC)->ARGS;

static void ArrayIntDivFullCPP(benchmark::State &state) {
    int count = static_cast<int>(state.range(0));
    int *src0 = new int[count];
    int *src1 = new int[count];
    int *dst = new int[count];
    int *mask = new int[count];
    init_random(src0, count);
    init_random(src1, count);
    init_full_mask(mask, count);

    for (auto _ : state) {
        array_int_div_full_cpp(src0, src1, dst, mask, count);
    }

    check(src0, src1, dst, mask, count, div_i);
    delete[] src0;
    delete[] src1;
    delete[] dst;
    delete[] mask;
}
BENCHMARK(ArrayIntDivFullCPP)->ARGS;

static void ArrayIntDivHalfISPC(benchmark::State &state) {
    int count = static_cast<int>(state.range(0));
    int *src0 = new int[count];
    int *src1 = new int[count];
    int *dst = new int[count];
    int *mask = new int[count];
    init_random(src0, count);
    init_random(src1, count);
    init_half_mask(mask, count);

    for (auto _ : state) {
        ispc::ArrayIntDivMask(src0, src1, dst, mask, count);
    }

    check(src0, src1, dst, mask, count, div_i_mask);
    delete[] src0;
    delete[] src1;
    delete[] dst;
    delete[] mask;
}
BENCHMARK(ArrayIntDivHalfISPC)->ARGS;

static void ArrayIntDivHalfCPP(benchmark::State &state) {
    int count = static_cast<int>(state.range(0));
    int *src0 = new int[count];
    int *src1 = new int[count];
    int *dst = new int[count];
    int *mask = new int[count];
    init_random(src0, count);
    init_random(src1, count);
    init_half_mask(mask, count);

    for (auto _ : state) {
        array_int_div_mask_cpp(src0, src1, dst, mask, count);
    }

    check(src0, src1, dst, mask, count, div_i_mask);
    delete[] src0;
    delete[] src1;
    delete[] dst;
    delete[] mask;
}
BENCHMARK(ArrayIntDivHalfCPP)->ARGS;

static void ArrayIntDivQuatISPC(benchmark::State &state) {
    int count = static_cast<int>(state.range(0));
    int *src0 = new int[count];
    int *src1 = new int[count];
    int *dst = new int[count];
    int *mask = new int[count];
    init_random(src0, count);
    init_random(src1, count);
    init_quat_mask(mask, count);

    for (auto _ : state) {
        ispc::ArrayIntDivMask(src0, src1, dst, mask, count);
    }

    check(src0, src1, dst, mask, count, div_i_mask);
    delete[] src0;
    delete[] src1;
    delete[] dst;
    delete[] mask;
}
BENCHMARK(ArrayIntDivQuatISPC)->ARGS;

static void ArrayIntDivQuatCPP(benchmark::State &state) {
    int count = static_cast<int>(state.range(0));
    int *src0 = new int[count];
    int *src1 = new int[count];
    int *dst = new int[count];
    int *mask = new int[count];
    init_random(src0, count);
    init_random(src1, count);
    init_quat_mask(mask, count);

    for (auto _ : state) {
        array_int_div_mask_cpp(src0, src1, dst, mask, count);
    }

    check(src0, src1, dst, mask, count, div_i_mask);
    delete[] src0;
    delete[] src1;
    delete[] dst;
    delete[] mask;
}
BENCHMARK(ArrayIntDivQuatCPP)->ARGS;

BENCHMARK_MAIN();
