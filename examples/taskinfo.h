/*
  Copyright (c) 2011, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  
*/

#ifndef TASKINFO_H
#define TASKINFO_H 1

#ifdef _MSC_VER
#define ISPC_IS_WINDOWS
#elif defined(__linux__)
#define ISPC_IS_LINUX
#elif defined(__APPLE__)
#define ISPC_IS_APPLE
#endif

#ifdef ISPC_IS_WINDOWS
#define NOMINMAX
#include <windows.h>
#include <concrt.h>
using namespace Concurrency;
#endif // ISPC_IS_WINDOWS

#if (__SIZEOF_POINTER__ == 4) || defined(__i386__) || defined(_WIN32)
#define ISPC_POINTER_BYTES 4
#elif (__SIZEOF_POINTER__ == 8) || defined(__x86_64__) || defined(__amd64__) || defined(_WIN64)
#define ISPC_POINTER_BYTES 8
#else
#error "Pointer size unknown!"
#endif // __SIZEOF_POINTER__

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

typedef struct TaskInfo {
    void *func;
    void *data;
#if defined(ISPC_IS_WINDOWS)
    event taskEvent;
#endif
} TaskInfo;


#ifndef ISPC_IS_WINDOWS
static int32_t 
lAtomicCompareAndSwap32(volatile int32_t *v, int32_t newValue, int32_t oldValue) {
    int32_t result;
    __asm__ __volatile__("lock\ncmpxchgl %2,%1"
                          : "=a"(result), "=m"(*v)
                          : "q"(newValue), "0"(oldValue)
                          : "memory");
    __asm__ __volatile__("mfence":::"memory");
    return result;
}
#endif // !ISPC_IS_WINDOWS


static void *
lAtomicCompareAndSwapPointer(void **v, void *newValue, void *oldValue) {
#ifdef ISPC_IS_WINDOWS
	return InterlockedCompareExchangePointer(v, newValue, oldValue);
#else
    void *result;
#if (ISPC_POINTER_BYTES == 4)
    __asm__ __volatile__("lock\ncmpxchgd %2,%1"
                          : "=a"(result), "=m"(*v)
                          : "q"(newValue), "0"(oldValue)
                          : "memory");
#else
    __asm__ __volatile__("lock\ncmpxchgq %2,%1"
                          : "=a"(result), "=m"(*v)
                          : "q"(newValue), "0"(oldValue)
                          : "memory");
#endif // ISPC_POINTER_BYTES
    __asm__ __volatile__("mfence":::"memory");
    return result;
#endif // ISPC_IS_WINDOWS
}


#ifndef ISPC_IS_WINDOWS
static int32_t 
lAtomicAdd32(volatile int32_t *v, int32_t delta) {
    // Do atomic add with gcc x86 inline assembly
    int32_t origValue;
    __asm__ __volatile__("lock\n"
                         "xaddl %0,%1"
                         : "=r"(origValue), "=m"(*v) : "0"(delta)
                         : "memory");
    return origValue;
}
#endif

#define LOG_TASK_QUEUE_CHUNK_SIZE 13
#define MAX_TASK_QUEUE_CHUNKS 1024
#define TASK_QUEUE_CHUNK_SIZE (1<<LOG_TASK_QUEUE_CHUNK_SIZE)

#define MAX_LAUNCHED_TASKS (MAX_TASK_QUEUE_CHUNKS * TASK_QUEUE_CHUNK_SIZE)

typedef void (*TaskFuncType)(void *, int, int);

#ifdef ISPC_IS_WINDOWS
static volatile LONG nextTaskInfoCoordinate;
#else
static volatile int nextTaskInfoCoordinate;
#endif

static TaskInfo *taskInfo[MAX_TASK_QUEUE_CHUNKS];

static inline void
lInitTaskInfo() {
    taskInfo[0] = new TaskInfo[TASK_QUEUE_CHUNK_SIZE];
}


static inline TaskInfo *
lGetTaskInfo() {
#ifdef ISPC_IS_WINDOWS
    int myCoord = InterlockedAdd(&nextTaskInfoCoordinate, 1)-1;
#else
    int myCoord = lAtomicAdd32(&nextTaskInfoCoordinate, 1);
#endif
	int index = (myCoord >> LOG_TASK_QUEUE_CHUNK_SIZE);
    int offset = myCoord & (TASK_QUEUE_CHUNK_SIZE-1);
    if (index == MAX_TASK_QUEUE_CHUNKS) {
        fprintf(stderr, "A total of %d tasks have been launched--the simple "
                "built-in task system can handle no more. Exiting.", myCoord);
        exit(1);
    }

    if (taskInfo[index] == NULL) {
        TaskInfo *newChunk = new TaskInfo[TASK_QUEUE_CHUNK_SIZE];
        if (lAtomicCompareAndSwapPointer((void **)&taskInfo[index], newChunk, 
                                         NULL) != NULL) {
            // failure--someone else got it, but that's cool
            assert(taskInfo[index] != NULL);
            free(newChunk);
        }
    }

    return &taskInfo[index][offset];
}


static inline void
lResetTaskInfo() {
    nextTaskInfoCoordinate = 0;
}

#endif // TASKINFO_H
