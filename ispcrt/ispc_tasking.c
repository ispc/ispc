// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <stdint.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Signature of ispc-generated 'task' functions
typedef void (*TaskFuncType)(void *data, int threadIndex, int threadCount, int taskIndex, int taskCount, int taskIndex0,
                             int taskIndex1, int taskIndex2, int taskCount0, int taskCount1, int taskCount2);

void ISPCLaunch(void **taskGroupPtr, void *_func, void *data, int count0, int count1, int count2) {
    const int count = count0 * count1 * count2;
    TaskFuncType func = (TaskFuncType)_func;

#pragma omp parallel
    {
#ifdef _OPENMP
        const int threadIndex = omp_get_thread_num();
        const int threadCount = omp_get_num_threads();
#else
        const int threadIndex = 0;
        const int threadCount = 1;
#endif
        int i = 0;
#pragma omp for schedule(runtime)
        for (i = 0; i < count; i++) {
            int taskIndex0 = i % count0;
            int taskIndex1 = (i / count0) % count1;
            int taskIndex2 = i / (count0 * count1);

            func(data, threadIndex, threadCount, i, count, taskIndex0, taskIndex1, taskIndex2, count0, count1, count2);
        }
    }
}

void ISPCSync(void *h) { free(h); }

void *ISPCAlloc(void **taskGroupPtr, int64_t size, int32_t alignment) {
    *taskGroupPtr = aligned_alloc(alignment, size);
    return *taskGroupPtr;
}
