/*
  Copyright (c) 2014, Intel Corporation
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


#define DBG(x)
#include <omp.h>
#include <malloc.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <algorithm>

// Signature of ispc-generated 'task' functions
typedef void (*TaskFuncType)(void *data, int threadIndex, int threadCount,
                             int taskIndex, int taskCount,
                             int taskIndex0, int taskIndex1, int taskIndex2,
                             int taskCount0, int taskCount1, int taskCount2);

// Small structure used to hold the data for each task
#ifdef _MSC_VER
__declspec(align(16))
#endif
struct TaskInfo {
    TaskFuncType func;
    void *data;
    int taskIndex;
    int taskCount3d[3];
#if defined(ISPC_IS_WINDOWS)
    event taskEvent;
#endif
    int taskCount() const { return taskCount3d[0]*taskCount3d[1]*taskCount3d[2]; }
    int taskIndex0() const
    {
      return taskIndex % taskCount3d[0];
    }
    int taskIndex1() const
    {
      return ( taskIndex / taskCount3d[0] ) % taskCount3d[1];
    }
    int taskIndex2() const
    {
      return taskIndex / ( taskCount3d[0]*taskCount3d[1] );
    }
    int taskCount0() const { return taskCount3d[0]; }
    int taskCount1() const { return taskCount3d[1]; }
    int taskCount2() const { return taskCount3d[2]; }
    TaskInfo() { assert(sizeof(TaskInfo) % 32 == 0); }
}
#ifndef _MSC_VER
__attribute__((aligned(32)));
#endif
;

// ispc expects these functions to have C linkage / not be mangled
extern "C" {
    void ISPCLaunch(void **handlePtr, void *f, void *data, int countx, int county, int countz);
    void *ISPCAlloc(void **handlePtr, int64_t size, int32_t alignment);
    void ISPCSync(void *handle);
}

///////////////////////////////////////////////////////////////////////////
// TaskGroupBase

#define LOG_TASK_QUEUE_CHUNK_SIZE 14
#define MAX_TASK_QUEUE_CHUNKS 8
#define TASK_QUEUE_CHUNK_SIZE (1<<LOG_TASK_QUEUE_CHUNK_SIZE)

#define MAX_LAUNCHED_TASKS (MAX_TASK_QUEUE_CHUNKS * TASK_QUEUE_CHUNK_SIZE)

#define NUM_MEM_BUFFERS 16

class TaskGroup;

/** The TaskGroupBase structure provides common functionality for "task
    groups"; a task group is the set of tasks launched from within a single
    ispc function.  When the function is ready to return, it waits for all
    of the tasks in its task group to finish before it actually returns.
 */
class TaskGroupBase {
public:
    void Reset();

    int AllocTaskInfo(int count);
    TaskInfo *GetTaskInfo(int index);

    void *AllocMemory(int64_t size, int32_t alignment);

protected:
    TaskGroupBase();
    ~TaskGroupBase();

    int nextTaskInfoIndex;

private:
    /* We allocate blocks of TASK_QUEUE_CHUNK_SIZE TaskInfo structures as
       needed by the calling function.  We hold up to MAX_TASK_QUEUE_CHUNKS
       of these (and then exit at runtime if more than this many tasks are
       launched.)
     */
    TaskInfo *taskInfo[MAX_TASK_QUEUE_CHUNKS];

    /* We also allocate chunks of memory to service ISPCAlloc() calls.  The
       memBuffers[] array holds pointers to this memory.  The first element
       of this array is initialized to point to mem and then any subsequent
       elements required are initialized with dynamic allocation.
     */
    int curMemBuffer, curMemBufferOffset;
    int memBufferSize[NUM_MEM_BUFFERS];
    char *memBuffers[NUM_MEM_BUFFERS];
    char mem[256];
};


inline TaskGroupBase::TaskGroupBase() {
    nextTaskInfoIndex = 0;

    curMemBuffer = 0;
    curMemBufferOffset = 0;
    memBuffers[0] = mem;
    memBufferSize[0] = sizeof(mem) / sizeof(mem[0]);
    for (int i = 1; i < NUM_MEM_BUFFERS; ++i) {
        memBuffers[i] = NULL;
        memBufferSize[i] = 0;
    }

    for (int i = 0; i < MAX_TASK_QUEUE_CHUNKS; ++i)
        taskInfo[i] = NULL;
}


inline TaskGroupBase::~TaskGroupBase() {
    // Note: don't delete memBuffers[0], since it points to the start of
    // the "mem" member!
    for (int i = 1; i < NUM_MEM_BUFFERS; ++i)
        delete[](memBuffers[i]);
}


inline void
TaskGroupBase::Reset() {
    nextTaskInfoIndex = 0;
    curMemBuffer = 0;
    curMemBufferOffset = 0;
}


inline int
TaskGroupBase::AllocTaskInfo(int count) {
    int ret = nextTaskInfoIndex;
    nextTaskInfoIndex += count;
    return ret;
}


inline TaskInfo *
TaskGroupBase::GetTaskInfo(int index) {
    int chunk = (index >> LOG_TASK_QUEUE_CHUNK_SIZE);
    int offset = index & (TASK_QUEUE_CHUNK_SIZE-1);

    if (chunk == MAX_TASK_QUEUE_CHUNKS) {
        fprintf(stderr, "A total of %d tasks have been launched from the "
                "current function--the simple built-in task system can handle "
                "no more. You can increase the values of TASK_QUEUE_CHUNK_SIZE "
                "and LOG_TASK_QUEUE_CHUNK_SIZE to work around this limitation.  "
                "Sorry!  Exiting.\n", index);
        exit(1);
    }

    if (taskInfo[chunk] == NULL)
        taskInfo[chunk] = new TaskInfo[TASK_QUEUE_CHUNK_SIZE];
    return &taskInfo[chunk][offset];
}


inline void *
TaskGroupBase::AllocMemory(int64_t size, int32_t alignment) {
    char *basePtr = memBuffers[curMemBuffer];
    intptr_t iptr = (intptr_t)(basePtr + curMemBufferOffset);
    iptr = (iptr + (alignment-1)) & ~(alignment-1);

    int newOffset = int(iptr - (intptr_t)basePtr + size);
    if (newOffset < memBufferSize[curMemBuffer]) {
        curMemBufferOffset = newOffset;
        return (char *)iptr;
    }

    ++curMemBuffer;
    curMemBufferOffset = 0;
    assert(curMemBuffer < NUM_MEM_BUFFERS);

    int allocSize = 1 << (12 + curMemBuffer);
    allocSize = std::max(int(size+alignment), allocSize);
    char *newBuf = new char[allocSize];
    memBufferSize[curMemBuffer] = allocSize;
    memBuffers[curMemBuffer] = newBuf;
    return AllocMemory(size, alignment);
}


///////////////////////////////////////////////////////////////////////////
// Atomics and the like

static inline void
lMemFence() {
    // Windows atomic functions already contain the fence
    // KNC doesn't need the memory barrier
#if !defined ISPC_IS_KNC && !defined ISPC_IS_WINDOWS
    __sync_synchronize();
#endif
}

static void *
lAtomicCompareAndSwapPointer(void **v, void *newValue, void *oldValue) {
#ifdef ISPC_IS_WINDOWS
    return InterlockedCompareExchangePointer(v, newValue, oldValue);
#else
    void *result = __sync_val_compare_and_swap(v, oldValue, newValue);
    lMemFence();
    return result;
#endif // ISPC_IS_WINDOWS
}

static int32_t
lAtomicCompareAndSwap32(volatile int32_t *v, int32_t newValue, int32_t oldValue) {
#ifdef ISPC_IS_WINDOWS
    return InterlockedCompareExchange((volatile LONG *)v, newValue, oldValue);
#else
    int32_t result = __sync_val_compare_and_swap(v, oldValue, newValue);
    lMemFence();
    return result;
#endif // ISPC_IS_WINDOWS
}

static inline int32_t
lAtomicAdd(volatile int32_t *v, int32_t delta) {
#ifdef ISPC_IS_WINDOWS
    return InterlockedExchangeAdd((volatile LONG *)v, delta)+delta;
#else
    return __sync_fetch_and_add(v, delta);
#endif
}

///////////////////////////////////////////////////////////////////////////

class TaskGroup : public TaskGroupBase {
public:
    void Launch(int baseIndex, int count);
    void Sync();

};


///////////////////////////////////////////////////////////////////////////
// OpenMP

static void
InitTaskSystem() {
        // No initialization needed
}

inline void
TaskGroup::Launch(int baseIndex, int count) {
#pragma omp parallel
  {
    const int threadIndex = omp_get_thread_num();
    const int threadCount = omp_get_num_threads();

    TaskInfo ti = *GetTaskInfo(baseIndex);
#pragma omp for schedule(runtime)
    for(int i = 0; i < count; i++)
    {
        ti.taskIndex = i;

        // Actually run the task.
        ti.func(ti.data, threadIndex, threadCount, ti.taskIndex, ti.taskCount(),
            ti.taskIndex0(), ti.taskIndex1(), ti.taskIndex2(),
            ti.taskCount0(), ti.taskCount1(), ti.taskCount2());
    }
  }
}

inline void
TaskGroup::Sync() {
}

///////////////////////////////////////////////////////////////////////////

#define MAX_FREE_TASK_GROUPS 64
static TaskGroup *freeTaskGroups[MAX_FREE_TASK_GROUPS];

  static inline TaskGroup *
AllocTaskGroup()
{
  for (int i = 0; i < MAX_FREE_TASK_GROUPS; ++i) {
    TaskGroup *tg = freeTaskGroups[i];
    if (tg != NULL) {
      void *ptr = lAtomicCompareAndSwapPointer((void **)(&freeTaskGroups[i]), NULL, tg);
      if (ptr != NULL) {
        return (TaskGroup *)ptr;
      }
    }
  }

  return new TaskGroup;
}


  static inline void
FreeTaskGroup(TaskGroup *tg)
{
  tg->Reset();

  for (int i = 0; i < MAX_FREE_TASK_GROUPS; ++i) {
    if (freeTaskGroups[i] == NULL) {
      void *ptr = lAtomicCompareAndSwapPointer((void **)&freeTaskGroups[i], tg, NULL);
      if (ptr == NULL)
        return;
    }
  }

  delete tg;
}

  void
ISPCLaunch(void **taskGroupPtr, void *func, void *data, int count0, int count1, int count2)
{
  const int count = count0*count1*count2;
  TaskGroup *taskGroup;
  if (*taskGroupPtr == NULL) {
    InitTaskSystem();
    taskGroup = AllocTaskGroup();
    *taskGroupPtr = taskGroup;
  }
  else
    taskGroup = (TaskGroup *)(*taskGroupPtr);

  int baseIndex = taskGroup->AllocTaskInfo(count);
  for (int i = 0; i < 1; ++i) {
    TaskInfo *ti = taskGroup->GetTaskInfo(baseIndex+i);
    ti->func = (TaskFuncType)func;
    ti->data = data;
    ti->taskIndex = i;
    ti->taskCount3d[0] = count0;
    ti->taskCount3d[1] = count1;
    ti->taskCount3d[2] = count2;
  }
  taskGroup->Launch(baseIndex, count);
}


  void
ISPCSync(void *h)
{
  TaskGroup *taskGroup = (TaskGroup *)h;
  if (taskGroup != NULL) {
    taskGroup->Sync();
    FreeTaskGroup(taskGroup);
  }
}


  void *
ISPCAlloc(void **taskGroupPtr, int64_t size, int32_t alignment)
{
  TaskGroup *taskGroup;
  if (*taskGroupPtr == NULL) {
    InitTaskSystem();
    taskGroup = AllocTaskGroup();
    *taskGroupPtr = taskGroup;
  }
  else
    taskGroup = (TaskGroup *)(*taskGroupPtr);

  return taskGroup->AllocMemory(size, alignment);
}

