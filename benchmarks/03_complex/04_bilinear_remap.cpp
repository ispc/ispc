// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdint>
#include <stdio.h>

#include "../common.h"
#include "04_bilinear_remap_ispc.h"

static Docs docs("Bilinear remap of an image through a grid of sample coordinates (discussion #2919):\n"
                 "remap - one channel (HW layout), remap_hwc - 3 interleaved channels (HWC layout).\n"
                 "Every output element reads 4 data-dependent image elements and the 2 adjacent coordinates of its\n"
                 "grid point (a stride-2 read).\n"
                 "remap_deinterleave, remap_hwc_deinterleave - references: the grid read as vector loads and a\n"
                 "deinterleave (aos_to_soa2).\n"
                 "The argument is the number of output pixels (a square image); the input image has the same size.\n"
                 "Expectation:\n"
                 " - No regressions\n"
                 " - remap as fast as remap_deinterleave: the stride-2 grid read needs no gather. The 4 image reads\n"
                 "   stay gathers, so the gain is small\n");

WARM_UP_RUN();

// Square images of 16 * 16, 64 * 64 and 256 * 256 pixels.
#define ARGS Arg(256)->Arg(4096)->Arg(65536)

static const int C = 3;

static uint64_t next_random(uint64_t &s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

// Sample coordinates inside the image and a margin of 1 around it (clamped by the kernel).
static void init(float *img, float *grid, int side, int channels) {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < side * side * channels; i++) {
        img[i] = static_cast<float>(next_random(s) % 1000) * 0.01f;
    }
    for (int i = 0; i < 2 * side * side; i++) {
        grid[i] = static_cast<float>(next_random(s) % (100 * (side + 2))) * 0.01f - 1.0f;
    }
}

static float ref_sample(const float *img, const float *grid, int side, int channels, int y, int x, int c) {
    float px_f = grid[(y * side + x) * 2 + 0];
    float py_f = grid[(y * side + x) * 2 + 1];
    int py = static_cast<int>(std::floor(py_f));
    int px = static_cast<int>(std::floor(px_f));
    float py_m = py_f - py;
    float px_m = px_f - px;
    auto clampi = [&](int v) { return v < 0 ? 0 : v > side - 1 ? side - 1 : v; };
    int px0 = clampi(px), px1 = clampi(px + 1), py0 = clampi(py), py1 = clampi(py + 1);
    auto at = [&](int yy, int xx) { return img[(yy * side + xx) * channels + c]; };
    return at(py0, px0) * (1.0f - py_m) * (1.0f - px_m) + at(py0, px1) * (1.0f - py_m) * px_m +
           at(py1, px0) * py_m * (1.0f - px_m) + at(py1, px1) * py_m * px_m;
}

static bool check(const float *img, const float *grid, const float *out, int side, int channels) {
    for (int y = 0; y < side; y++) {
        for (int x = 0; x < side; x++) {
            for (int c = 0; c < channels; c++) {
                float want = ref_sample(img, grid, side, channels, y, x, c);
                float got = out[(y * side + x) * channels + c];
                if (!(std::abs(got - want) <= 1e-4f * (1.0f + std::abs(want)))) {
                    printf("Error y=%d x=%d c=%d: %f, expected %f\n", y, x, c, got, want);
                    return false;
                }
            }
        }
    }
    return true;
}

// CALL is the call of the kernel, with the channel count in the HWC layout.
#define BENCH(NAME, CHANNELS, CALL)                                                                                    \
    static void NAME(benchmark::State &state) {                                                                        \
        int side = static_cast<int>(std::lround(std::sqrt(static_cast<double>(state.range(0)))));                      \
        float *img = static_cast<float *>(aligned_alloc_helper(sizeof(float) * side * side * CHANNELS));               \
        float *grid = static_cast<float *>(aligned_alloc_helper(sizeof(float) * side * side * 2));                     \
        float *out = static_cast<float *>(aligned_alloc_helper(sizeof(float) * side * side * CHANNELS));               \
        init(img, grid, side, CHANNELS);                                                                               \
        for (int i = 0; i < side * side * CHANNELS; i++) {                                                             \
            out[i] = NAN;                                                                                              \
        }                                                                                                              \
        for (auto _ : state) {                                                                                         \
            CALL;                                                                                                      \
        }                                                                                                              \
        if (!check(img, grid, out, side, CHANNELS))                                                                    \
            fail_benchmark(state);                                                                                     \
        aligned_free_helper(img);                                                                                      \
        aligned_free_helper(grid);                                                                                     \
        aligned_free_helper(out);                                                                                      \
        state.SetComplexityN(state.range(0));                                                                          \
    }                                                                                                                  \
    BENCHMARK(NAME)->ARGS;

BENCH(remap, 1, ispc::remap(out, img, grid, side, side, side, side))
BENCH(remap_deinterleave, 1, ispc::remap_deinterleave(out, img, grid, side, side, side, side))
BENCH(remap_hwc, C, ispc::remap_hwc(out, img, grid, side, side, C, side, side))
BENCH(remap_hwc_deinterleave, C, ispc::remap_hwc_deinterleave(out, img, grid, side, side, C, side, side))

CHECKED_BENCHMARK_MAIN();
