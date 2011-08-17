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

#include "taskinfo.h"

/* Simple task system implementation for ispc based on Microsoft's
   Concurrency Runtime. */

#include <windows.h>
#include <concrt.h>
using namespace Concurrency;
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

// ispc expects these functions to have C linkage / not be mangled
extern "C" { 
    void ISPCLaunch(void *f, void *data);
    void ISPCSync();
    void *ISPCMalloc(int64_t size, int32_t alignment);
    void ISPCFree(void *ptr);
}


void __cdecl
lRunTask(LPVOID param) {
    TaskInfo *ti = (TaskInfo *)param;
    
    // Actually run the task. 
    // FIXME: like the GCD implementation for OS X, this is passing bogus
    // values for the threadIndex and threadCount builtins, which in turn
    // will cause bugs in code that uses those.
    int threadIndex = 0;
    int threadCount = 1;
    TaskFuncType func = (TaskFuncType)ti->func;
    func(ti->data, threadIndex, threadCount);

    // Signal the event that this task is done
    ti->taskEvent.set();
}


void
ISPCLaunch(void *func, void *data) {
    TaskInfo *ti = lGetTaskInfo();
    ti->func = (TaskFuncType)func;
    ti->data = data;
	ti->taskEvent.reset();
    CurrentScheduler::ScheduleTask(lRunTask, ti);
}


void ISPCSync() {
    for (int i = 0; i < nextTaskInfoCoordinate; ++i) {
		int index = (i >> LOG_TASK_QUEUE_CHUNK_SIZE);
		int offset = i & (TASK_QUEUE_CHUNK_SIZE-1);
		taskInfo[index][offset].taskEvent.wait();
		taskInfo[index][offset].taskEvent.reset();
    }

    lResetTaskInfo();
}


void *ISPCMalloc(int64_t size, int32_t alignment) {
    return _aligned_malloc(size, alignment);
}


void ISPCFree(void *ptr) {
    _aligned_free(ptr);
}
