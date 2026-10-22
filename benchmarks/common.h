// Copyright (c) 2021-2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "config.h"

// Set maximum alignment for existing ISPC targets.
#define ALIGNMENT 64

// Granularity of 4K aliasing between loads and stores.
#define ALIASING_PAGE_SIZE 4096

class Docs {
  public:
    Docs(std::string message) {
        std::cout << "BENCHMARKS_ISPC_TARGETS: " << BENCHMARKS_ISPC_TARGETS << "\n";
        std::cout << "BENCHMARKS_ISPC_FLAGS: " << BENCHMARKS_ISPC_FLAGS << "\n";
        std::cout << message << "\n";
    }
};

// Set when a correctness check of any benchmark failed.
inline bool benchmark_check_failed = false;

// Reports wrong results of the running benchmark: it is shown with an error, and the executable
// exits with a non-zero code (see CHECKED_BENCHMARK_MAIN).
inline void fail_benchmark(benchmark::State &state) {
    benchmark_check_failed = true;
    state.SkipWithError("Incorrect results");
}

// Same as BENCHMARK_MAIN(), but returns 1 if a correctness check failed. The ISPC targets of the
// executable are added to the context of the results.
#define CHECKED_BENCHMARK_MAIN()                                                                                       \
    int main(int argc, char **argv) {                                                                                  \
        benchmark::MaybeReenterWithoutASLR(argc, argv);                                                                \
        benchmark::Initialize(&argc, argv);                                                                            \
        if (benchmark::ReportUnrecognizedArguments(argc, argv))                                                        \
            return 1;                                                                                                  \
        benchmark::AddCustomContext("ispc_targets", BENCHMARKS_ISPC_TARGETS);                                          \
        benchmark::RunSpecifiedBenchmarks();                                                                           \
        benchmark::Shutdown();                                                                                         \
        return benchmark_check_failed ? 1 : 0;                                                                         \
    }

// Number of buffer slots in a page and the mask of slots taken by live aligned_alloc_helper() buffers.
#define BUFFER_SLOTS (ALIASING_PAGE_SIZE / ALIGNMENT)
inline uint64_t used_buffer_slots = 0;

// Helper function to allocate aligned buffers.
// Performance of many kernels depends on the distance between their buffers modulo the page size
// (4K aliasing of loads from one buffer with stores to another one), so placing buffers wherever
// malloc() puts them makes results depend on the heap state, i.e. on everything allocated before.
// To make the placement deterministic, a buffer starts at a fixed offset from a page boundary: the
// first free slot in bit-reversed order, i.e. 0, 2048, 1024, 3072, 512, ... for 4K pages. So buffers
// allocated together are as far from each other as possible.
inline void *aligned_alloc_helper(size_t size, size_t alignment = ALIGNMENT) {
    if (ALIGNMENT % alignment != 0) {
        std::cerr << "aligned_alloc_helper: alignment " << alignment << " is bigger than " << ALIGNMENT << std::endl;
        std::abort();
    }
    static_assert(BUFFER_SLOTS <= 64, "Slots don't fit in used_buffer_slots");
    unsigned slot = 0;
    for (unsigned n = 0; n < BUFFER_SLOTS; n++) {
        slot = 0;
        for (unsigned bit = 1, rest = n; bit < BUFFER_SLOTS; bit <<= 1, rest >>= 1) {
            slot = (slot << 1) | (rest & 1);
        }
        if (!(used_buffer_slots & (uint64_t(1) << slot))) {
            break;
        }
    }
    // Buffer start and the malloc() pointer and slot stored right before it fit in 2 extra pages.
    char *raw = static_cast<char *>(malloc(size + 2 * ALIASING_PAGE_SIZE));
    if (raw == nullptr) {
        return nullptr;
    }
    used_buffer_slots |= uint64_t(1) << slot;
    uintptr_t page = (reinterpret_cast<uintptr_t>(raw) + 2 * sizeof(void *) + ALIASING_PAGE_SIZE - 1) &
                     ~uintptr_t(ALIASING_PAGE_SIZE - 1);
    void **ptr = reinterpret_cast<void **>(page + slot * ALIGNMENT);
    ptr[-1] = raw;
    ptr[-2] = reinterpret_cast<void *>(uintptr_t(slot));
    return ptr;
}

inline void aligned_free_helper(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    void **header = static_cast<void **>(ptr);
    used_buffer_slots &= ~(uint64_t(1) << reinterpret_cast<uintptr_t>(header[-2]));
    free(header[-1]);
}

// Warm up run
// Despite other ways of CPU stabilization, on some CPUSs we need to do a "warm up" in order to get more stable results.
// This also helps stabilizing results when other ways to fix frequency are not available / not applicable.
#define WARM_UP_RUN()                                                                                                  \
    static void warm_up(benchmark::State &state) {                                                                     \
        int count = static_cast<int>(state.range(0));                                                                  \
        float *src = static_cast<float *>(aligned_alloc_helper(sizeof(float) * count));                                \
        float *dst = static_cast<float *>(aligned_alloc_helper(sizeof(float) * count));                                \
        for (int i = 0; i < count; i++) {                                                                              \
            src[i] = 1.0f;                                                                                             \
            dst[i] = 0.0f;                                                                                             \
        }                                                                                                              \
                                                                                                                       \
        for (auto _ : state) {                                                                                         \
            for (int i = 0; i < count; i++) {                                                                          \
                benchmark::DoNotOptimize(dst[i] = src[i] * 2.0f + 3.0f);                                               \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        aligned_free_helper(src);                                                                                      \
        aligned_free_helper(dst);                                                                                      \
    }                                                                                                                  \
    BENCHMARK(warm_up)->Arg(256)->Arg(256 << 4)->Arg(256 << 7)->Arg(256 << 12);
