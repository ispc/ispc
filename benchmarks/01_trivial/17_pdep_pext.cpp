// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <benchmark/benchmark.h>
#include <cstdint>
#include <stdio.h>

#include "../common.h"
#include "17_pdep_pext_ispc.h"

static Docs docs("Check pdep32/pext32/pdep64/pext64 implementation of stdlib functions:\n"
                 "[pdep, pext] x [uint32, uint64] x [uniform, varying] versions with sparse and dense runtime masks,\n"
                 "full and partial execution masks, and the same masks as compile-time constants and runtime values.\n"
                 "Morton encode/decode with pdep/pext (constant and runtime masks) compared to shift/AND/OR.\n"
                 "Expectation:\n"
                 " - No regressions\n"
                 " - Native BMI2 instructions on AVX2 and AVX-512 targets for runtime masks\n"
                 " - Varying operations with constant masks at least as fast as with runtime masks\n");

WARM_UP_RUN();

// Second argument: 0 for sparse masks, 1 for dense masks.
#define ARGS Args({8192, 0})->Args({8192, 1})

static uint64_t next_random(uint64_t &s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

template <typename T> static void init(T *src, T *mask, T *dst, int count, bool dense) {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < count; i++) {
        src[i] = static_cast<T>(next_random(s));
        uint64_t a = next_random(s);
        uint64_t b = next_random(s);
        mask[i] = static_cast<T>(dense ? (a | b) : (a & b & next_random(s)));
        dst[i] = 0;
    }
}

template <typename T> static T ref_pdep(T value, T mask) {
    T result = 0;
    for (T bit = 1; mask != 0; bit <<= 1) {
        T lowest = mask & (T(0) - mask);
        if ((value & bit) != 0)
            result |= lowest;
        mask &= mask - 1;
    }
    return result;
}

template <typename T> static T ref_pext(T value, T mask) {
    T result = 0;
    for (T bit = 1; mask != 0; bit <<= 1) {
        T lowest = mask & (T(0) - mask);
        if ((value & lowest) != 0)
            result |= bit;
        mask &= mask - 1;
    }
    return result;
}

// Elements with (src[i] % stride) != 0 are skipped and must keep dst[i] == 0.
template <typename T>
static void check(const T *src, const T *mask, const T *dst, int count, bool is_pdep, int stride) {
    for (int i = 0; i < count; ++i) {
        T expected = 0;
        if (src[i] % stride == 0)
            expected = is_pdep ? ref_pdep(src[i], mask[i]) : ref_pext(src[i], mask[i]);
        if (expected != dst[i]) {
            printf("Error i=%d expected=%llx result=%llx\n", i, (unsigned long long)expected,
                   (unsigned long long)dst[i]);
            return;
        }
    }
}

#define BENCHMARK_BIT_OP(OP, W, V, STRIDE, IS_PDEP)                                                                    \
    static void OP##W##_##V(benchmark::State &state) {                                                                 \
        int count = static_cast<int>(state.range(0));                                                                  \
        bool dense = state.range(1) != 0;                                                                              \
        uint##W##_t *src = static_cast<uint##W##_t *>(aligned_alloc_helper(sizeof(uint##W##_t) * count));              \
        uint##W##_t *mask = static_cast<uint##W##_t *>(aligned_alloc_helper(sizeof(uint##W##_t) * count));             \
        uint##W##_t *dst = static_cast<uint##W##_t *>(aligned_alloc_helper(sizeof(uint##W##_t) * count));              \
        init(src, mask, dst, count, dense);                                                                            \
                                                                                                                       \
        for (auto _ : state) {                                                                                         \
            ispc::OP##W##_##V(src, mask, dst, count);                                                                  \
        }                                                                                                              \
                                                                                                                       \
        check(src, mask, dst, count, IS_PDEP, STRIDE);                                                                 \
        aligned_free_helper(src);                                                                                      \
        aligned_free_helper(mask);                                                                                     \
        aligned_free_helper(dst);                                                                                      \
        state.SetComplexityN(state.range(0));                                                                          \
    }                                                                                                                  \
    BENCHMARK(OP##W##_##V)->ARGS;

#define BENCHMARK_BIT_OPS(OP, W, IS_PDEP)                                                                              \
    BENCHMARK_BIT_OP(OP, W, uniform, 1, IS_PDEP)                                                                       \
    BENCHMARK_BIT_OP(OP, W, varying, 1, IS_PDEP)                                                                       \
    BENCHMARK_BIT_OP(OP, W, varying_half, 2, IS_PDEP)                                                                  \
    BENCHMARK_BIT_OP(OP, W, varying_sparse, 8, IS_PDEP)

BENCHMARK_BIT_OPS(pdep, 32, true)
BENCHMARK_BIT_OPS(pext, 32, false)
BENCHMARK_BIT_OPS(pdep, 64, true)
BENCHMARK_BIT_OPS(pext, 64, false)

// The same mask as a compile-time constant (CONST) and as a runtime argument (RUNTIME).
#define BENCHMARK_MASK_OP_KIND(OP, W, NAME, MASK, V, IS_PDEP, KIND, ...)                                               \
    static void OP##W##_##NAME##_##KIND##_##V(benchmark::State &state) {                                               \
        int count = static_cast<int>(state.range(0));                                                                  \
        uint##W##_t *src = static_cast<uint##W##_t *>(aligned_alloc_helper(sizeof(uint##W##_t) * count));              \
        uint##W##_t *mask = static_cast<uint##W##_t *>(aligned_alloc_helper(sizeof(uint##W##_t) * count));             \
        uint##W##_t *dst = static_cast<uint##W##_t *>(aligned_alloc_helper(sizeof(uint##W##_t) * count));              \
        init(src, mask, dst, count, false);                                                                            \
        for (int i = 0; i < count; i++) {                                                                              \
            mask[i] = MASK;                                                                                            \
        }                                                                                                              \
                                                                                                                       \
        for (auto _ : state) {                                                                                         \
            ispc::OP##W##_##NAME##_##KIND##_##V(src, dst, count __VA_ARGS__);                                          \
        }                                                                                                              \
                                                                                                                       \
        check(src, mask, dst, count, IS_PDEP, 1);                                                                      \
        aligned_free_helper(src);                                                                                      \
        aligned_free_helper(mask);                                                                                     \
        aligned_free_helper(dst);                                                                                      \
        state.SetComplexityN(state.range(0));                                                                          \
    }                                                                                                                  \
    BENCHMARK(OP##W##_##NAME##_##KIND##_##V)->Arg(8192);

#define BENCHMARK_MASK_OP(OP, W, NAME, MASK, IS_PDEP)                                                                  \
    BENCHMARK_MASK_OP_KIND(OP, W, NAME, MASK, uniform, IS_PDEP, const, )                                               \
    BENCHMARK_MASK_OP_KIND(OP, W, NAME, MASK, uniform, IS_PDEP, runtime, , MASK)                                       \
    BENCHMARK_MASK_OP_KIND(OP, W, NAME, MASK, varying, IS_PDEP, const, )                                               \
    BENCHMARK_MASK_OP_KIND(OP, W, NAME, MASK, varying, IS_PDEP, runtime, , MASK)

BENCHMARK_MASK_OP(pdep, 32, morton, 0x55555555u, true)
BENCHMARK_MASK_OP(pext, 32, morton, 0x55555555u, false)
BENCHMARK_MASK_OP(pdep, 64, morton, 0x5555555555555555ull, true)
BENCHMARK_MASK_OP(pext, 64, morton, 0x5555555555555555ull, false)
BENCHMARK_MASK_OP(pdep, 32, bytes, 0x00FF00FFu, true)
BENCHMARK_MASK_OP(pext, 32, bytes, 0x00FF00FFu, false)
BENCHMARK_MASK_OP(pdep, 64, bytes, 0x00FF00FF00FF00FFull, true)
BENCHMARK_MASK_OP(pext, 64, bytes, 0x00FF00FF00FF00FFull, false)

static uint64_t ref_morton(uint32_t x, uint32_t y) {
    return ref_pdep<uint64_t>(x, 0x5555555555555555ull) | ref_pdep<uint64_t>(y, 0xAAAAAAAAAAAAAAAAull);
}

#define BENCHMARK_MORTON_ENCODE(IMPL, ...)                                                                             \
    static void morton_encode_##IMPL(benchmark::State &state) {                                                        \
        int count = static_cast<int>(state.range(0));                                                                  \
        uint32_t *x = static_cast<uint32_t *>(aligned_alloc_helper(sizeof(uint32_t) * count));                         \
        uint32_t *y = static_cast<uint32_t *>(aligned_alloc_helper(sizeof(uint32_t) * count));                         \
        uint64_t *dst = static_cast<uint64_t *>(aligned_alloc_helper(sizeof(uint64_t) * count));                       \
        uint64_t s = 0x9E3779B97F4A7C15ull;                                                                            \
        for (int i = 0; i < count; i++) {                                                                              \
            x[i] = static_cast<uint32_t>(next_random(s));                                                              \
            y[i] = static_cast<uint32_t>(next_random(s));                                                              \
            dst[i] = 0;                                                                                                \
        }                                                                                                              \
                                                                                                                       \
        for (auto _ : state) {                                                                                         \
            ispc::morton_encode_##IMPL(x, y, dst, count __VA_ARGS__);                                                  \
        }                                                                                                              \
                                                                                                                       \
        for (int i = 0; i < count; ++i) {                                                                              \
            if (dst[i] != ref_morton(x[i], y[i])) {                                                                    \
                printf("Error i=%d\n", i);                                                                             \
                break;                                                                                                 \
            }                                                                                                          \
        }                                                                                                              \
        aligned_free_helper(x);                                                                                        \
        aligned_free_helper(y);                                                                                        \
        aligned_free_helper(dst);                                                                                      \
        state.SetComplexityN(state.range(0));                                                                          \
    }                                                                                                                  \
    BENCHMARK(morton_encode_##IMPL)->Arg(8192);

#define BENCHMARK_MORTON_DECODE(IMPL, ...)                                                                             \
    static void morton_decode_##IMPL(benchmark::State &state) {                                                        \
        int count = static_cast<int>(state.range(0));                                                                  \
        uint64_t *src = static_cast<uint64_t *>(aligned_alloc_helper(sizeof(uint64_t) * count));                       \
        uint32_t *x = static_cast<uint32_t *>(aligned_alloc_helper(sizeof(uint32_t) * count));                         \
        uint32_t *y = static_cast<uint32_t *>(aligned_alloc_helper(sizeof(uint32_t) * count));                         \
        uint64_t s = 0x9E3779B97F4A7C15ull;                                                                            \
        for (int i = 0; i < count; i++) {                                                                              \
            src[i] = next_random(s);                                                                                   \
            x[i] = 0;                                                                                                  \
            y[i] = 0;                                                                                                  \
        }                                                                                                              \
                                                                                                                       \
        for (auto _ : state) {                                                                                         \
            ispc::morton_decode_##IMPL(src, x, y, count __VA_ARGS__);                                                  \
        }                                                                                                              \
                                                                                                                       \
        for (int i = 0; i < count; ++i) {                                                                              \
            if (ref_morton(x[i], y[i]) != src[i]) {                                                                    \
                printf("Error i=%d\n", i);                                                                             \
                break;                                                                                                 \
            }                                                                                                          \
        }                                                                                                              \
        aligned_free_helper(src);                                                                                      \
        aligned_free_helper(x);                                                                                        \
        aligned_free_helper(y);                                                                                        \
        state.SetComplexityN(state.range(0));                                                                          \
    }                                                                                                                  \
    BENCHMARK(morton_decode_##IMPL)->Arg(8192);

BENCHMARK_MORTON_ENCODE(pdep_const, )
BENCHMARK_MORTON_ENCODE(pdep_runtime, , 0x5555555555555555ull, 0xAAAAAAAAAAAAAAAAull)
BENCHMARK_MORTON_ENCODE(shifts, )
BENCHMARK_MORTON_DECODE(pext_const, )
BENCHMARK_MORTON_DECODE(pext_runtime, , 0x5555555555555555ull, 0xAAAAAAAAAAAAAAAAull)
BENCHMARK_MORTON_DECODE(shifts, )

BENCHMARK_MAIN();
