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

/* A simple task system for ispc programs based on Apple's Grand Central
   Dispatch. */

#include <dispatch/dispatch.h>
#include <stdio.h>
#include <stdlib.h>

static bool initialized = false;
static dispatch_queue_t gcdQueue;
static dispatch_group_t gcdGroup;

// ispc expects these functions to have C linkage / not be mangled
extern "C" {
    void ISPCLaunch(void *f, void *data);
    void ISPCSync();
}

struct TaskInfo {
    void *func;
    void *data;
};


void
TasksInit() {
    gcdQueue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
    gcdGroup = dispatch_group_create();
    initialized = true;
}


static void
lRunTask(void *ti) {
    typedef void (*TaskFuncType)(void *, int, int);
    TaskInfo *taskInfo = (TaskInfo *)ti;

    TaskFuncType func = (TaskFuncType)(taskInfo->func);

    // FIXME: these are bogus values; may cause bugs in code that depends
    // on them having unique values in different threads.
    int threadIndex = 0;
    int threadCount = 1;
    // Actually run the task
    func(taskInfo->data, threadIndex, threadCount);

    // FIXME: taskInfo leaks...
}


void ISPCLaunch(void *func, void *data) {
    if (!initialized) {
        fprintf(stderr, "You must call TasksInit() before launching tasks.\n");
        exit(1);
    }
    TaskInfo *ti = new TaskInfo;
    ti->func = func;
    ti->data = data;
    dispatch_group_async_f(gcdGroup, gcdQueue, ti, lRunTask);
}


void ISPCSync() {
    if (!initialized) {
        fprintf(stderr, "You must call TasksInit() before launching tasks.\n");
        exit(1);
    }

    // Wait for all of the tasks in the group to complete before returning
    dispatch_group_wait(gcdGroup, DISPATCH_TIME_FOREVER);
}
