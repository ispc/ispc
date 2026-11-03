// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <benchmark/benchmark.h>
#include <cstdint>
#include <stdio.h>

#include "../common.h"
#include "18_gather_scatter_ispc.h"

static Docs docs("Gathers and scatters of int8, int32 and double elements:\n"
                 "{gather,scatter}_random_<element>_<index>: through an index array (a random permutation) of type\n"
                 "int32, uint32, int16 or uint16;\n"
                 "{gather,scatter}_random_masked_<element>_<index>: the same with half of the lanes active;\n"
                 "{gather,scatter}_random_tail_<element>_<index>: the same with 4093 elements, so that the last\n"
                 "iteration is partial;\n"
                 "{gather,scatter}_{broadcast4,reverse,stride2,stride4}_<element>: index patterns computed in the\n"
                 "kernel (i & ~3, n - 1 - i, 2 * i, 4 * i), which the compiler can turn into vector loads, stores and\n"
                 "shuffles.\n"
                 "Expectation:\n"
                 " - No regressions\n"
                 " - Unsigned and 16-bit indices as fast as int32 indices where the index range allows a dword\n"
                 "   gather or scatter\n"
                 " - Visible patterns faster than random ones\n");

WARM_UP_RUN();

// The int16 indices limit the size to 32768; the uint16 indices go up to 65535.
#define ARGS Arg(256)->Arg(4096)->Arg(32768)
#define U16_ARGS Arg(256)->Arg(4096)->Arg(65536)
#define TAIL_ARGS Arg(4093)

// Elements after the end of every buffer, which must keep their initial value.
#define PAD 64

static uint64_t next_random(uint64_t &s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

// A random permutation of [0, count).
template <typename IT> static void init_permutation(IT *idx, int count) {
    for (int i = 0; i < count; i++) {
        idx[i] = static_cast<IT>(i);
    }
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (int i = count - 1; i > 0; i--) {
        int j = static_cast<int>(next_random(s) % static_cast<uint64_t>(i + 1));
        IT t = idx[i];
        idx[i] = idx[j];
        idx[j] = t;
    }
}

// Distinct values for int32 and double; for int8 a hash of the index, so that equal values are not a fixed number
// of elements apart. All values differ from the initial value -1 of the destinations.
template <typename ET> static ET value(int i) {
    if (sizeof(ET) == 1) {
        return static_cast<ET>((static_cast<uint32_t>(i) * 2654435761u) >> 25);
    }
    return static_cast<ET>(3 * i + 1);
}

// Fills src_count + PAD and dst_count + PAD elements.
template <typename ET> static void init_values(ET *src, ET *dst, int src_count, int dst_count) {
    for (int i = 0; i < src_count + PAD; i++) {
        src[i] = value<ET>(i);
    }
    for (int i = 0; i < dst_count + PAD; i++) {
        dst[i] = static_cast<ET>(-1);
    }
}

// The PAD elements after dst_count are untouched.
template <typename ET> static bool check_pad(const ET *dst, int dst_count) {
    for (int i = dst_count; i < dst_count + PAD; i++) {
        if (dst[i] != static_cast<ET>(-1)) {
            printf("Error: element %d after the end was written\n", i);
            return false;
        }
    }
    return true;
}

static void init_half_on(int8_t *on, int count) {
    for (int i = 0; i < count; i++) {
        on[i] = static_cast<int8_t>((i * 7 / 3) % 2);
    }
}

// NAME is random or random_tail.
#define BENCH_RANDOM(OP, NAME, ET_C, ET, IT_C, IT, ARG_LIST)                                                           \
    static void OP##_##NAME##_##ET##_##IT(benchmark::State &state) {                                                   \
        int count = static_cast<int>(state.range(0));                                                                  \
        ET_C *src = static_cast<ET_C *>(aligned_alloc_helper(sizeof(ET_C) * (count + PAD)));                           \
        ET_C *dst = static_cast<ET_C *>(aligned_alloc_helper(sizeof(ET_C) * (count + PAD)));                           \
        IT_C *idx = static_cast<IT_C *>(aligned_alloc_helper(sizeof(IT_C) * count));                                   \
        init_values(src, dst, count, count);                                                                           \
        init_permutation(idx, count);                                                                                  \
        for (auto _ : state) {                                                                                         \
            ispc::OP##_random_##ET##_##IT(src, idx, dst, count);                                                       \
        }                                                                                                              \
        for (int i = 0; i < count; i++) {                                                                              \
            bool ok = #OP[0] == 'g' ? dst[i] == src[idx[i]] : dst[idx[i]] == src[i];                                   \
            if (!ok) {                                                                                                 \
                printf("Error i=%d\n", i);                                                                             \
                fail_benchmark(state);                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
        }                                                                                                              \
        if (!check_pad(dst, count))                                                                                    \
            fail_benchmark(state);                                                                                     \
        aligned_free_helper(src);                                                                                      \
        aligned_free_helper(dst);                                                                                      \
        aligned_free_helper(idx);                                                                                      \
        state.SetComplexityN(state.range(0));                                                                          \
    }                                                                                                                  \
    BENCHMARK(OP##_##NAME##_##ET##_##IT)->ARG_LIST;

#define BENCH_RANDOM_MASKED(OP, ET_C, ET, IT_C, IT)                                                                    \
    static void OP##_random_masked_##ET##_##IT(benchmark::State &state) {                                              \
        int count = static_cast<int>(state.range(0));                                                                  \
        ET_C *src = static_cast<ET_C *>(aligned_alloc_helper(sizeof(ET_C) * (count + PAD)));                           \
        ET_C *dst = static_cast<ET_C *>(aligned_alloc_helper(sizeof(ET_C) * (count + PAD)));                           \
        IT_C *idx = static_cast<IT_C *>(aligned_alloc_helper(sizeof(IT_C) * count));                                   \
        int8_t *on = static_cast<int8_t *>(aligned_alloc_helper(count));                                               \
        init_values(src, dst, count, count);                                                                           \
        init_permutation(idx, count);                                                                                  \
        init_half_on(on, count);                                                                                       \
        for (auto _ : state) {                                                                                         \
            ispc::OP##_random_masked_##ET##_##IT(src, idx, on, dst, count);                                            \
        }                                                                                                              \
        for (int i = 0; i < count; i++) {                                                                              \
            int d = #OP[0] == 'g' ? i : static_cast<int>(idx[i]);                                                      \
            ET_C want = on[i] ? (#OP[0] == 'g' ? src[idx[i]] : src[i]) : static_cast<ET_C>(-1);                        \
            if (dst[d] != want) {                                                                                      \
                printf("Error i=%d\n", i);                                                                             \
                fail_benchmark(state);                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
        }                                                                                                              \
        if (!check_pad(dst, count))                                                                                    \
            fail_benchmark(state);                                                                                     \
        aligned_free_helper(src);                                                                                      \
        aligned_free_helper(dst);                                                                                      \
        aligned_free_helper(idx);                                                                                      \
        aligned_free_helper(on);                                                                                       \
        state.SetComplexityN(state.range(0));                                                                          \
    }                                                                                                                  \
    BENCHMARK(OP##_random_masked_##ET##_##IT)->ARGS;

// dst[i] = src[f(i)] for a gather, dst[f(i)] = src[i] for a scatter. SPAN is the size of the strided array. For a
// scatter, every other element of dst, including the gaps of a stride, must keep its initial value.
#define BENCH_PATTERN(OP, PATTERN, ET_C, ET, F, SPAN)                                                                  \
    static void OP##_##PATTERN##_##ET(benchmark::State &state) {                                                       \
        int count = static_cast<int>(state.range(0));                                                                  \
        int gather = #OP[0] == 'g';                                                                                    \
        int src_count = count * (gather ? SPAN : 1), dst_count = count * (gather ? 1 : SPAN);                          \
        ET_C *src = static_cast<ET_C *>(aligned_alloc_helper(sizeof(ET_C) * (src_count + PAD)));                       \
        ET_C *dst = static_cast<ET_C *>(aligned_alloc_helper(sizeof(ET_C) * (dst_count + PAD)));                       \
        ET_C *want = static_cast<ET_C *>(aligned_alloc_helper(sizeof(ET_C) * dst_count));                              \
        init_values(src, dst, src_count, dst_count);                                                                   \
        for (int j = 0; j < dst_count; j++) {                                                                          \
            want[j] = static_cast<ET_C>(-1);                                                                           \
        }                                                                                                              \
        for (int i = 0; i < count; i++) {                                                                              \
            int n = count;                                                                                             \
            (void)n;                                                                                                   \
            int f = F;                                                                                                 \
            if (gather)                                                                                                \
                want[i] = src[f];                                                                                      \
            else                                                                                                       \
                want[f] = src[i];                                                                                      \
        }                                                                                                              \
        for (auto _ : state) {                                                                                         \
            ispc::OP##_##PATTERN##_##ET(src, dst, count);                                                              \
        }                                                                                                              \
        for (int j = 0; j < dst_count; j++) {                                                                          \
            if (dst[j] != want[j]) {                                                                                   \
                printf("Error j=%d\n", j);                                                                             \
                fail_benchmark(state);                                                                                 \
                break;                                                                                                 \
            }                                                                                                          \
        }                                                                                                              \
        if (!check_pad(dst, dst_count))                                                                                \
            fail_benchmark(state);                                                                                     \
        aligned_free_helper(src);                                                                                      \
        aligned_free_helper(dst);                                                                                      \
        aligned_free_helper(want);                                                                                     \
        state.SetComplexityN(state.range(0));                                                                          \
    }                                                                                                                  \
    BENCHMARK(OP##_##PATTERN##_##ET)->ARGS;

BENCH_RANDOM(gather, random, int32_t, int32, int32_t, int32, ARGS)
BENCH_RANDOM(gather, random, int32_t, int32, uint32_t, uint32, ARGS)
BENCH_RANDOM(gather, random, int32_t, int32, int16_t, int16, ARGS)
BENCH_RANDOM(gather, random, int32_t, int32, uint16_t, uint16, U16_ARGS)
BENCH_RANDOM(gather, random, int8_t, int8, int32_t, int32, ARGS)
BENCH_RANDOM(gather, random, int8_t, int8, uint32_t, uint32, ARGS)
BENCH_RANDOM(gather, random, double, double, int32_t, int32, ARGS)
BENCH_RANDOM(gather, random, double, double, uint32_t, uint32, ARGS)
BENCH_RANDOM(gather, random_tail, int32_t, int32, int32_t, int32, TAIL_ARGS)

BENCH_RANDOM(scatter, random, int32_t, int32, int32_t, int32, ARGS)
BENCH_RANDOM(scatter, random, int32_t, int32, uint32_t, uint32, ARGS)
BENCH_RANDOM(scatter, random, int32_t, int32, int16_t, int16, ARGS)
BENCH_RANDOM(scatter, random, int8_t, int8, int32_t, int32, ARGS)
BENCH_RANDOM(scatter, random, double, double, int32_t, int32, ARGS)
BENCH_RANDOM(scatter, random_tail, int32_t, int32, int32_t, int32, TAIL_ARGS)

BENCH_RANDOM_MASKED(gather, int32_t, int32, int32_t, int32)
BENCH_RANDOM_MASKED(gather, double, double, int32_t, int32)
BENCH_RANDOM_MASKED(scatter, int32_t, int32, int32_t, int32)

BENCH_PATTERN(gather, broadcast4, int8_t, int8, i & ~3, 1)
BENCH_PATTERN(gather, broadcast4, int32_t, int32, i & ~3, 1)
BENCH_PATTERN(gather, broadcast4, double, double, i & ~3, 1)
BENCH_PATTERN(gather, reverse, int8_t, int8, n - 1 - i, 1)
BENCH_PATTERN(gather, reverse, int32_t, int32, n - 1 - i, 1)
BENCH_PATTERN(gather, reverse, double, double, n - 1 - i, 1)
BENCH_PATTERN(gather, stride2, int8_t, int8, 2 * i, 2)
BENCH_PATTERN(gather, stride2, int32_t, int32, 2 * i, 2)
BENCH_PATTERN(gather, stride2, double, double, 2 * i, 2)
BENCH_PATTERN(gather, stride4, int8_t, int8, 4 * i, 4)
BENCH_PATTERN(gather, stride4, int32_t, int32, 4 * i, 4)
BENCH_PATTERN(gather, stride4, double, double, 4 * i, 4)

BENCH_PATTERN(scatter, reverse, int32_t, int32, n - 1 - i, 1)
BENCH_PATTERN(scatter, reverse, double, double, n - 1 - i, 1)
BENCH_PATTERN(scatter, stride2, int32_t, int32, 2 * i, 2)
BENCH_PATTERN(scatter, stride2, double, double, 2 * i, 2)
BENCH_PATTERN(scatter, stride4, int32_t, int32, 4 * i, 4)
BENCH_PATTERN(scatter, stride4, double, double, 4 * i, 4)

CHECKED_BENCHMARK_MAIN();
