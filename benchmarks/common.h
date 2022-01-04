#include <cassert>
#include <iostream>
#include <string>

#if defined(_WIN32) || defined(_WIN64)
#include <malloc.h>
#else
#include <cstdlib>
#endif

// Set maximum alignment for existing ISPC targets.
#define ALIGNMENT 64

class Docs {
  public:
    Docs(std::string message) {
        std::cout << "BENCHMARKS_ISPC_TARGETS: " << BENCHMARKS_ISPC_TARGETS << "\n";
        std::cout << "BENCHMARKS_ISPC_FLAGS: " << BENCHMARKS_ISPC_FLAGS << "\n";
        std::cout << message << "\n";
    }
};

// Helper function to enabled allocated allocations.
// Despite aligned_alloc() is part of C++, MSVS doesn't support it.
void *aligned_alloc_helper(size_t size, size_t alignment = ALIGNMENT) {
    assert(size % alignment == 0 && "Size is not multiple of alignment");
#if defined(_WIN32) || defined(_WIN64)
    return _aligned_malloc(size, alignment);
#else
    return aligned_alloc(alignment, size);
#endif
}

void aligned_free_helper(void *ptr) {
#if defined(_WIN32) || defined(_WIN64)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
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

