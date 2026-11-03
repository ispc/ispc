// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstring>
#include <stdio.h>

#include "../common.h"
#include "03_rgba8_diff_ispc.h"

static Docs docs("Difference of two RGBA8 images (discussion #2943): the pixels that differ get a marker color.\n"
                 "difference_rgba8 - the original kernel, byte accesses at index * 4 + c\n"
                 "difference_rgba8_u32 - reference: the same with one uint32 per pixel\n"
                 "Expectation:\n"
                 " - No regressions\n"
                 " - difference_rgba8 as fast as difference_rgba8_u32: the 4 bytes of a pixel are contiguous,\n"
                 "   so the byte gathers and scatters can be vector loads and a masked store\n");

WARM_UP_RUN();

#define ARGS Arg(256)->Arg(4096)->Arg(65536)

static uint64_t next_random(uint64_t &s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

// About a quarter of the pixels differ, in one channel.
static void init(uint8_t *src, uint8_t *dst, uint8_t *out, int count) {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < count; i++) {
        uint64_t r = next_random(s);
        for (int c = 0; c < 4; c++) {
            src[4 * i + c] = static_cast<uint8_t>(r >> (8 * c));
            dst[4 * i + c] = src[4 * i + c];
            out[4 * i + c] = 17;
        }
        if ((r >> 32) % 4 == 0) {
            dst[4 * i + (r >> 40) % 4] ^= 1;
        }
    }
}

static bool check(const uint8_t *src, const uint8_t *dst, const uint8_t *out, int count) {
    static const uint8_t marker[4] = {255, 0, 255, 255};
    for (int i = 0; i < count; i++) {
        bool differs = memcmp(src + 4 * i, dst + 4 * i, 4) != 0;
        for (int c = 0; c < 4; c++) {
            if (out[4 * i + c] != (differs ? marker[c] : 17)) {
                printf("Error i=%d\n", i);
                return false;
            }
        }
    }
    return true;
}

#define BENCH(NAME, T)                                                                                                 \
    static void NAME(benchmark::State &state) {                                                                        \
        int count = static_cast<int>(state.range(0));                                                                  \
        uint8_t *src = static_cast<uint8_t *>(aligned_alloc_helper(4 * count));                                        \
        uint8_t *dst = static_cast<uint8_t *>(aligned_alloc_helper(4 * count));                                        \
        uint8_t *out = static_cast<uint8_t *>(aligned_alloc_helper(4 * count));                                        \
        init(src, dst, out, count);                                                                                    \
        for (auto _ : state) {                                                                                         \
            ispc::NAME(reinterpret_cast<T *>(src), reinterpret_cast<T *>(dst), reinterpret_cast<T *>(out), count);     \
        }                                                                                                              \
        if (!check(src, dst, out, count))                                                                              \
            fail_benchmark(state);                                                                                     \
        aligned_free_helper(src);                                                                                      \
        aligned_free_helper(dst);                                                                                      \
        aligned_free_helper(out);                                                                                      \
        state.SetComplexityN(state.range(0));                                                                          \
    }                                                                                                                  \
    BENCHMARK(NAME)->ARGS;

BENCH(difference_rgba8, uint8_t)
BENCH(difference_rgba8_u32, uint32_t)

CHECKED_BENCHMARK_MAIN();
