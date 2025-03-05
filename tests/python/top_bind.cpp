// This file defines functions required for the launch functionality used in the functional tests.
// All functions taken from $ISPC_HOME/tests/test_static.cpp

#include <malloc.h>
#include <stdint.h>

extern "C" {
void ISPCLaunch(void **handlePtr, void *f, void *d, int, int, int);
void ISPCSync(void *handle);
void *ISPCAlloc(void **handlePtr, int64_t size, int32_t alignment);
}

void ISPCLaunch(void **handle, void *f, void *d, int count0, int count1, int count2) {
    *handle = (void *)(uintptr_t)0xdeadbeef;
    typedef void (*TaskFuncType)(void *, int, int, int, int, int, int, int, int, int, int);
    TaskFuncType func = (TaskFuncType)f;
    int count = count0 * count1 * count2, idx = 0;
    for (int k = 0; k < count2; ++k)
        for (int j = 0; j < count1; ++j)
            for (int i = 0; i < count0; ++i)
                func(d, 0, 1, idx++, count, i, j, k, count0, count1, count2);
}

void ISPCSync(void *) {}

void *ISPCAlloc(void **handle, int64_t size, int32_t alignment) {
    *handle = (void *)(uintptr_t)0xdeadbeef;
    // TODO: Handle different platforms. Assumes you are on Linux
    // and now, we leak...
    // #ifdef ISPC_IS_WINDOWS
    //     return _aligned_malloc(size, alignment);
    // #elif defined ISPC_IS_LINUX
    return memalign(alignment, size);
    // #elif defined ISPC_IS_APPLE || defined ISPC_IS_WASM
    //     void *mem = malloc(size + (alignment - 1) + sizeof(void *));
    //     char *amem = ((char *)mem) + sizeof(void *);
    //     amem = amem + uint32_t(alignment - (reinterpret_cast<uint64_t>(amem) & (alignment - 1)));
    //     ((void **)amem)[-1] = mem;
    //     return amem;
    // #else
    // #error "Host OS was not detected"
    // #endif
}
