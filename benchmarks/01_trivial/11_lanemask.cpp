// Copyright (c) 2025, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <benchmark/benchmark.h>
#include <cstdint>
#include <stdio.h>

#include "../common.h"
#include "11_lanemask_ispc.h"

static Docs docs("Check performance of conditional lanemask() function.\n"
                 "Things to note:\n"
                 " - workload sizes are designed to hit different caches (L1/L2/L3).\n"
                 "Expectations:\n"
                 " - No regressions\n");

WARM_UP_RUN();

// Minimum size is maximum target width * 4, i.e. 64*4 = 256.
// 256 * sizeof(uint64_t) = 2kb - expected to reside in L1
// 256 * sizeof(uint64_t) << 4 = 32kb - expected to reside in L1/L2
// 256 * sizeof(uint64_t) << 7 = 256kb - expected to reside in L2
// 256 * sizeof(uint64_t) << 12 = 8Mb - expected to reside in L3
#define ARGS Arg(256)->Arg(256 << 4)->Arg(256 << 7)->Arg(256 << 12)

static void init_src(int *src, int count) {
    // Using xorshift for fast, simple, non-repeating pattern of bits
    uint32_t x = 0x12345678;

    for (int i = 0; i < count; i++) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        if ((x >> 31) & 1) {
            src[i] = -1;
        } else {
            src[i] = 0;
        }
    }
}

static void check_lanemask(int *src, uint64_t *dst, int count, int programCount) {
    for (int i = 0, k = 0; i < count; i += programCount, k++) {
        uint64_t mask = 0;
        for (int j = 0; j < programCount; j++) {
            if (src[i + j]) {
                mask |= 1 << j;
            }
        }
        if (mask != dst[k]) {
            printf("Mismatch at index %d: expected %016lx, got %016lx\n", k, mask, dst[k]);
            return;
        }
    }
}

static void lanemask_test(benchmark::State &state) {
    int count = static_cast<int>(state.range(0));
    int *src = static_cast<int *>(aligned_alloc_helper(sizeof(int) * count));
    uint64_t *dst = static_cast<uint64_t *>(aligned_alloc_helper(sizeof(uint64_t) * count));

    init_src(src, count);

    for (auto _ : state) {
        ispc::lanemask_test(src, dst, count);
    }

    // Get programCount from the target width
    check_lanemask(src, dst, count, ispc::get_ispc_program_count());

    aligned_free_helper(src);
    aligned_free_helper(dst);
}
BENCHMARK(lanemask_test)->ARGS;

BENCHMARK_MAIN();
