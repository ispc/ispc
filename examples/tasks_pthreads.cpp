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
#include <pthread.h>
#include <semaphore.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/param.h>
#include <sys/sysctl.h>
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>

static int initialized = 0;
static volatile int32_t lock = 0;

static int nThreads;
static pthread_t *threads;
static pthread_mutex_t taskQueueMutex;
static int nextTaskToRun;
static sem_t *workerSemaphore;
static uint32_t numUnfinishedTasks;
static pthread_mutex_t tasksRunningConditionMutex;
static pthread_cond_t tasksRunningCondition;

// ispc expects these functions to have C linkage / not be mangled
extern "C" { 
    void ISPCLaunch(void *f, void *data);
    void ISPCSync();
}

static void *lTaskEntry(void *arg);

/** Figure out how many CPU cores there are in the system
 */
static int
lNumCPUCores() {
    return sysconf(_SC_NPROCESSORS_ONLN);
}


static void
lTasksInit() {
    nThreads = lNumCPUCores();

    threads = (pthread_t *)malloc(nThreads * sizeof(pthread_t));

    int err;
    if ((err = pthread_mutex_init(&taskQueueMutex, NULL)) != 0) {
        fprintf(stderr, "Error creating mutex: %s\n", strerror(err));
        exit(1);
    }

    char name[32];
    sprintf(name, "ispc_task.%d", (int)getpid());
    workerSemaphore = sem_open(name, O_CREAT, S_IRUSR|S_IWUSR, 0);
    if (!workerSemaphore) {
        fprintf(stderr, "Error creating semaphore: %s\n", strerror(err));
        exit(1);
    }

    if ((err = pthread_cond_init(&tasksRunningCondition, NULL)) != 0) {
        fprintf(stderr, "Error creating condition variable: %s\n", strerror(err));
        exit(1);
    }

    if ((err = pthread_mutex_init(&tasksRunningConditionMutex, NULL)) != 0) {
        fprintf(stderr, "Error creating mutex: %s\n", strerror(err));
        exit(1);
    }

    for (int i = 0; i < nThreads; ++i) {
        err = pthread_create(&threads[i], NULL, &lTaskEntry, (void *)(i));
        if (err != 0) {
            fprintf(stderr, "Error creating pthread %d: %s\n", i, strerror(err));
            exit(1);
        }
    }
}


void
ISPCLaunch(void *f, void *d) {
    int err;

    if (!initialized) {
        while (1) {
            if (lAtomicCompareAndSwap32(&lock, 1, 0) == 0) {
                if (!initialized) {
                    lTasksInit();
                    __asm__ __volatile__("mfence":::"memory");
                    initialized = 1;
                }
                lock = 0;
                break;
            }
        }
    }

    //
    // Acquire mutex, add task
    //
    if ((err = pthread_mutex_lock(&taskQueueMutex)) != 0) {
        fprintf(stderr, "Error from pthread_mutex_lock: %s\n", strerror(err));
        exit(1);
    }

    // Need a mutex here to ensure we get this filled in before a worker
    // grabs it and starts running...
    TaskInfo *ti = lGetTaskInfo();
    ti->func = f;
    ti->data = d;

    if ((err = pthread_mutex_unlock(&taskQueueMutex)) != 0) {
        fprintf(stderr, "Error from pthread_mutex_unlock: %s\n", strerror(err));
        exit(1);
    }

    //
    // Update count of number of tasks left to run
    //
    if ((err = pthread_mutex_lock(&tasksRunningConditionMutex)) != 0) {
        fprintf(stderr, "Error from pthread_mutex_lock: %s\n", strerror(err));
        exit(1);
    }

    // FIXME: is this redundant with nextTaskInfoCoordinate?
    ++numUnfinishedTasks;

    if ((err = pthread_mutex_unlock(&tasksRunningConditionMutex)) != 0) {
        fprintf(stderr, "Error from pthread_mutex_lock: %s\n", strerror(err));
        exit(1);
    }

    //
    // Post to the worker semaphore to wake up worker threads that are
    // sleeping waiting for tasks to show up
    //
    if ((err = sem_post(workerSemaphore)) != 0) {
        fprintf(stderr, "Error from sem_post: %s\n", strerror(err));
        exit(1);
    }
}


static void *
lTaskEntry(void *arg) {
    int threadIndex = (int)((int64_t)arg);
    int threadCount = nThreads;
    TaskFuncType func;

    while (1) {
        int err;
        if ((err = sem_wait(workerSemaphore)) != 0) {
            fprintf(stderr, "Error from sem_wait: %s\n", strerror(err));
            exit(1);
        }

        //
        // Acquire mutex, get task
        //
        if ((err = pthread_mutex_lock(&taskQueueMutex)) != 0) {
            fprintf(stderr, "Error from pthread_mutex_lock: %s\n", strerror(err));
            exit(1);
        }

        if (nextTaskToRun == nextTaskInfoCoordinate) {
            //
            // Task queue is empty, go back and wait on the semaphore
            //
            if ((err = pthread_mutex_unlock(&taskQueueMutex)) != 0) {
                fprintf(stderr, "Error from pthread_mutex_unlock: %s\n", strerror(err));
                exit(1);
            }
            continue;
        }

        int runCoord = nextTaskToRun++;
        int index = (runCoord >> LOG_TASK_QUEUE_CHUNK_SIZE);
        int offset = runCoord & (TASK_QUEUE_CHUNK_SIZE-1);
        TaskInfo *myTask = &taskInfo[index][offset];

        if ((err = pthread_mutex_unlock(&taskQueueMutex)) != 0) {
            fprintf(stderr, "Error from pthread_mutex_unlock: %s\n", strerror(err));
            exit(1);
        }

        //
        // Do work for _myTask_
        //
        func = (TaskFuncType)myTask->func;
        func(myTask->data, threadIndex, threadCount);

        //
        // Decrement the number of unfinished tasks counter
        //
        if ((err = pthread_mutex_lock(&tasksRunningConditionMutex)) != 0) {
            fprintf(stderr, "Error from pthread_mutex_lock: %s\n", strerror(err));
            exit(1);
        }

        // FIXME: can this be a comparison of (nextTaskToRun == nextTaskInfoCoordinate)?
        // (I don't think so--think there is a race...)
        int unfinished = --numUnfinishedTasks;
        if (unfinished == 0) {
            //
            // Signal the "no more tasks are running" condition if all of
            // them are done.
            //
            int err;
            if ((err = pthread_cond_signal(&tasksRunningCondition)) != 0) {
                fprintf(stderr, "Error from pthread_cond_signal: %s\n", strerror(err));
                exit(1);
            }
        }

        if ((err = pthread_mutex_unlock(&tasksRunningConditionMutex)) != 0) {
            fprintf(stderr, "Error from pthread_mutex_lock: %s\n", strerror(err));
            exit(1);
        }
    }

    pthread_exit(NULL);
    return 0;
}


void ISPCSync() {
    int err;
    if ((err = pthread_mutex_lock(&tasksRunningConditionMutex)) != 0) {
        fprintf(stderr, "Error from pthread_mutex_lock: %s\n", strerror(err));
        exit(1);
    }

    // As long as there are tasks running, wait on the condition variable;
    // doing so causes this thread to go to sleep until someone signals on
    // the tasksRunningCondition condition variable.
    while (numUnfinishedTasks > 0) {
        if ((err = pthread_cond_wait(&tasksRunningCondition, 
                                     &tasksRunningConditionMutex)) != 0) {
            fprintf(stderr, "Error from pthread_cond_wait: %s\n", strerror(err));
            exit(1);
        }
    }
    
    lResetTaskInfo();
    nextTaskToRun = 0;

    // We acquire ownership of the condition variable mutex when the above
    // pthread_cond_wait returns.
    // FIXME: is there a lurking issue here if numUnfinishedTasks gets back
    // to zero by the time we get to ISPCSync() and thence we're trying to
    // unlock a mutex we don't have a lock on?
    if ((err = pthread_mutex_unlock(&tasksRunningConditionMutex)) != 0) {
        fprintf(stderr, "Error from pthread_mutex_lock: %s\n", strerror(err));
        exit(1);
    }
}
