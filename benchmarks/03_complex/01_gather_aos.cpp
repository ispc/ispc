// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdint>
#include <stdio.h>

#include "../common.h"
#include "01_gather_aos_ispc.h"

static Docs
    docs("Moments (the sums of the coordinates and of the squared norms) of point records of N = 2, 3, 4 floats\n"
         "in an array of structures, read packed (record i), strided (record 2 * i) or indexed (record idx[i],\n"
         "random records):\n"
         "moments_gather_<way>_<N> - a varying struct access (gathers)\n"
         "moments_transpose_<way>_<N> - aos_to_soa<N> of contiguous records (of records copied lane by lane for\n"
         "the indexed way)\n"
         "moments_wide_<way>_<N> - the coordinates read as one contiguous float array (packed and strided)\n"
         "Expectation:\n"
         " - No regressions\n"
         " - Packed and strided gathers as fast as the transposes (coalesced into vector loads); the indexed\n"
         "   gathers stay genuine gathers\n");

WARM_UP_RUN();

// Number of records the gang reads.
#define ARGS Arg(256)->Arg(4096)->Arg(16384)

static uint64_t next_random(uint64_t &s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

static void init(float *f, int floats, int *idx, int n) {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < floats; i++) {
        f[i] = static_cast<float>(next_random(s) % 2000) * 0.001f - 1.0f;
    }
    // Random records, with repeats and omissions, so that a kernel that ignores idx gets other moments.
    for (int i = 0; i < n; i++) {
        idx[i] = static_cast<int>(next_random(s) % static_cast<uint64_t>(n));
    }
}

// Compares out with the moments of the records rec(0 .. n - 1) of f.
template <typename Rec> static bool check(const float *f, int n, int N, const float *out, Rec rec) {
    double s[4] = {0, 0, 0, 0}, q = 0;
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < N; k++) {
            double v = f[rec(i) * N + k];
            s[k] += v;
            q += v * v;
        }
    }
    for (int k = 0; k <= N; k++) {
        double want = k < N ? s[k] : q;
        if (!(std::abs(out[k] - want) <= 1e-3 * (1.0 + std::abs(want)))) {
            printf("Error k=%d: %f, expected %f\n", k, out[k], want);
            return false;
        }
    }
    return true;
}

// WAY is packed, strided or indexed; strided reads every other record of 2 * n records.
#define BENCH(VARIANT, WAY, N)                                                                                         \
    static void moments_##VARIANT##_##WAY##_##N(benchmark::State &state) {                                             \
        int n = static_cast<int>(state.range(0));                                                                      \
        bool strided = #WAY[0] == 's', indexed = #WAY[0] == 'i';                                                       \
        int records = strided ? 2 * n : n;                                                                             \
        float *f = static_cast<float *>(aligned_alloc_helper(sizeof(float) * N * records));                            \
        int *idx = static_cast<int *>(aligned_alloc_helper(sizeof(int) * n));                                          \
        float out[N + 1];                                                                                              \
        for (int k = 0; k <= N; k++) {                                                                                 \
            out[k] = NAN;                                                                                              \
        }                                                                                                              \
        init(f, N * records, idx, n);                                                                                  \
        auto *pts = reinterpret_cast<ispc::point##N *>(f);                                                             \
        for (auto _ : state) {                                                                                         \
            CALL_##WAY(VARIANT, N);                                                                                    \
        }                                                                                                              \
        bool ok = check(f, n, N, out, [&](int i) { return strided ? 2 * i : indexed ? idx[i] : i; });                  \
        if (!ok)                                                                                                       \
            fail_benchmark(state);                                                                                     \
        aligned_free_helper(f);                                                                                        \
        aligned_free_helper(idx);                                                                                      \
        state.SetComplexityN(state.range(0));                                                                          \
    }                                                                                                                  \
    BENCHMARK(moments_##VARIANT##_##WAY##_##N)->ARGS;

#define CALL_packed(VARIANT, N) ispc::moments_##VARIANT##_packed_##N(pts, n, out)
#define CALL_strided(VARIANT, N) ispc::moments_##VARIANT##_strided_##N(pts, n, out)
#define CALL_indexed(VARIANT, N) ispc::moments_##VARIANT##_indexed_##N(pts, idx, n, out)

#define BENCH_N(N)                                                                                                     \
    BENCH(gather, packed, N)                                                                                           \
    BENCH(transpose, packed, N)                                                                                        \
    BENCH(wide, packed, N)                                                                                             \
    BENCH(gather, strided, N)                                                                                          \
    BENCH(transpose, strided, N)                                                                                       \
    BENCH(wide, strided, N)                                                                                            \
    BENCH(gather, indexed, N)                                                                                          \
    BENCH(transpose, indexed, N)

BENCH_N(2)
BENCH_N(3)
BENCH_N(4)

CHECKED_BENCHMARK_MAIN();
