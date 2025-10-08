// Copyright (c) 2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <benchmark/benchmark.h>
#include <cstdint>
#include <stdio.h>

#include "../common.h"
#include "16_dispatcher_overhead_ispc.h"

static Docs docs("Measure ISPC dispatcher overhead with minimal computation.\n");

WARM_UP_RUN();

#define ARGS Arg(256)->Arg(256 << 4)->Arg(256 << 7)->Arg(256 << 9)

namespace ispc {

static void dispatcher_overhead_test() {

    const int M = 100;
    const int N = 100;

    int A[M][N];
    int B[M][N];
    int C[M][N];

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = add(i, j);

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            B[i][j] = sub(i, j);

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            C[i][j] = mul(A[i][j], B[i][j]);

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            assert(C[i][j] == (i + j) * (i - j));
}

} // namespace ispc

static void dispatcher_overhead_test(benchmark::State &state) {

    int count = static_cast<int>(state.range(0));
    for (auto _ : state)
        for (int i = 0; i < count; i++)
            ispc::dispatcher_overhead_test();

    state.SetComplexityN(state.range(0));
}

BENCHMARK(dispatcher_overhead_test)->ARGS;

BENCHMARK_MAIN();
