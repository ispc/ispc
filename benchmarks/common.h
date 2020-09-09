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
    Docs(std::string message) { std::cout << message << "\n"; }
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
