// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstring>
#include <stdio.h>

#include "../common.h"
#include "05_struct_copy_ispc.h"

static Docs
    docs("Copies of structs of byte fields (#2778): 8, 12 and 16 bytes, contiguous, indexed (half of the structs,\n"
         "chosen at random) and masked (half of the elements). References: the same copies through int64 (8 bytes),\n"
         "a struct of 3 int32 (12 bytes) and a struct of 2 int64 (16 bytes).\n"
         "Expectation:\n"
         " - No regressions\n"
         " - Byte struct copies as fast as the references: one 8- or 16-byte access per lane (or 3 dwords),\n"
         "   keeping the per-lane mask\n");

WARM_UP_RUN();

#define ARGS Arg(256)->Arg(4096)->Arg(65536)

static uint64_t next_random(uint64_t &s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

// The buffers hold 2 * count structs. The indexed copy copies count of them, chosen at random (so a kernel that
// ignores idx copies other structs); the contiguous and masked copies use the first count structs.
static void init(uint8_t *src, uint8_t *dst, int *idx, int8_t *on, int count, int size) {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < 2 * count * size; i++) {
        src[i] = static_cast<uint8_t>(next_random(s));
        dst[i] = 0xee;
    }
    int *perm = new int[2 * count];
    for (int i = 0; i < 2 * count; i++) {
        perm[i] = i;
    }
    for (int i = 2 * count - 1; i > 0; i--) {
        int j = static_cast<int>(next_random(s) % static_cast<uint64_t>(i + 1));
        int t = perm[i];
        perm[i] = perm[j];
        perm[j] = t;
    }
    for (int i = 0; i < count; i++) {
        idx[i] = perm[i];
        on[i] = static_cast<int8_t>((i * 7 / 3) % 2);
    }
    delete[] perm;
}

// Struct k of dst must be a copy of struct k of src if copied[k], else all bytes 0xee.
static bool check(const uint8_t *src, const uint8_t *dst, const bool *copied, int count, int size) {
    for (int k = 0; k < 2 * count; k++) {
        for (int b = 0; b < size; b++) {
            uint8_t want = copied[k] ? src[k * size + b] : 0xee;
            if (dst[k * size + b] != want) {
                printf("Error k=%d\n", k);
                return false;
            }
        }
    }
    return true;
}

// KIND is copy, copy_indexed or copy_masked.
#define BENCH(KIND, NAME, T, SIZE)                                                                                     \
    static void KIND##_##NAME(benchmark::State &state) {                                                               \
        int count = static_cast<int>(state.range(0));                                                                  \
        uint8_t *src = static_cast<uint8_t *>(aligned_alloc_helper(2 * count * SIZE));                                 \
        uint8_t *dst = static_cast<uint8_t *>(aligned_alloc_helper(2 * count * SIZE));                                 \
        int *idx = static_cast<int *>(aligned_alloc_helper(sizeof(int) * count));                                      \
        int8_t *on = static_cast<int8_t *>(aligned_alloc_helper(count));                                               \
        bool *copied = new bool[2 * count]();                                                                          \
        init(src, dst, idx, on, count, SIZE);                                                                          \
        bool masked = strcmp(#KIND, "copy_masked") == 0, indexed = strcmp(#KIND, "copy_indexed") == 0;                 \
        for (int i = 0; i < count; i++) {                                                                              \
            copied[indexed ? idx[i] : i] = !masked || on[i];                                                           \
        }                                                                                                              \
        for (auto _ : state) {                                                                                         \
            if (masked)                                                                                                \
                ispc::copy_masked_##NAME(reinterpret_cast<T *>(dst), reinterpret_cast<T *>(src), on, count);           \
            else if (indexed)                                                                                          \
                ispc::copy_indexed_##NAME(reinterpret_cast<T *>(dst), reinterpret_cast<T *>(src), idx, count);         \
            else                                                                                                       \
                ispc::copy_##NAME(reinterpret_cast<T *>(dst), reinterpret_cast<T *>(src), count);                      \
        }                                                                                                              \
        if (!check(src, dst, copied, count, SIZE))                                                                     \
            fail_benchmark(state);                                                                                     \
        delete[] copied;                                                                                               \
        aligned_free_helper(src);                                                                                      \
        aligned_free_helper(dst);                                                                                      \
        aligned_free_helper(idx);                                                                                      \
        aligned_free_helper(on);                                                                                       \
        state.SetComplexityN(state.range(0));                                                                          \
    }                                                                                                                  \
    BENCHMARK(KIND##_##NAME)->ARGS;

#define BENCH_ALL(NAME, T, SIZE)                                                                                       \
    BENCH(copy, NAME, T, SIZE)                                                                                         \
    BENCH(copy_indexed, NAME, T, SIZE)                                                                                 \
    BENCH(copy_masked, NAME, T, SIZE)

BENCH_ALL(bytes8, ispc::Bytes8, 8)
BENCH_ALL(int64, int64_t, 8)
BENCH_ALL(bytes12, ispc::Bytes12, 12)
BENCH_ALL(int32x3, ispc::Int32x3, 12)
BENCH_ALL(bytes16, ispc::Bytes16, 16)
BENCH_ALL(int64x2, ispc::Int64x2, 16)

CHECKED_BENCHMARK_MAIN();
