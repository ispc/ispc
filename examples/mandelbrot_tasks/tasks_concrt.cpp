/*
  Copyright (c) 2010-2011, Intel Corporation
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

/* Simple task system implementation for ispc based on Microsoft's
   Concurrency Runtime. */

#include <windows.h>
#include <concrt.h>
using namespace Concurrency;
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// ispc expects these functions to have C linkage / not be mangled
extern "C" { 
    void ISPCLaunch(void *f, void *data);
    void ISPCSync();
    void *ISPCMalloc(int64_t size, int32_t alignment);
    void ISPCFree(void *ptr);
}

typedef void (*TaskFuncType)(void *, int, int);

struct TaskInfo {
    TaskFuncType ispcFunc;
    void *ispcData;
};

// This is a simple implementation that just aborts if more than MAX_TASKS
// are launched.  It could easily be extended to be more general...

#define MAX_TASKS 4096
static int taskOffset;
static TaskInfo taskInfo[MAX_TASKS];
static event *events[MAX_TASKS];
static CRITICAL_SECTION criticalSection;
static bool initialized = false;

void
TasksInit() {
    InitializeCriticalSection(&criticalSection);
    for (int i = 0; i < MAX_TASKS; ++i)
        events[i] = new event;
    initialized = true;
}


void __cdecl
lRunTask(LPVOID param) {
    TaskInfo *ti = (TaskInfo *)param;
    
    // Actually run the task. 
    // FIXME: like the tasks_gcd.cpp implementation, this is passing bogus
    // values for the threadIndex and threadCount builtins, which in turn
    // will cause bugs in code that uses those.  FWIW this example doesn't
    // use them...
    int threadIndex = 0;
    int threadCount = 1;
    ti->ispcFunc(ti->ispcData, threadIndex, threadCount);

    // Signal the event that this task is done
    int taskNum = ti - &taskInfo[0];
    events[taskNum]->set();
}


void
ISPCLaunch(void *func, void *data) {
    if (!initialized) {
        fprintf(stderr, "You must call TasksInit() before launching tasks.\n");
        exit(1);
    }

    // Get a TaskInfo struct for this task
    EnterCriticalSection(&criticalSection);
    TaskInfo *ti = &taskInfo[taskOffset++];
    assert(taskOffset < MAX_TASKS);
    LeaveCriticalSection(&criticalSection);

    // And pass it on to the Concurrency Runtime...
    ti->ispcFunc = (TaskFuncType)func;
    ti->ispcData = data;
    CurrentScheduler::ScheduleTask(lRunTask, ti);
}


void ISPCSync() {
    if (!initialized) {
        fprintf(stderr, "You must call TasksInit() before launching tasks.\n");
        exit(1);
    }

    event::wait_for_multiple(&events[0], taskOffset, true, 
                             COOPERATIVE_TIMEOUT_INFINITE);

    for (int i = 0; i < taskOffset; ++i)
        events[i]->reset();

    taskOffset = 0;
}


void *ISPCMalloc(int64_t size, int32_t alignment) {
    return _aligned_malloc(size, alignment);
}


void ISPCFree(void *ptr) {
    _aligned_free(ptr);
}
