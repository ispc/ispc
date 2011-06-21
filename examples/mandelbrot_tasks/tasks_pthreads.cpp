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
#include <vector>

// ispc expects these functions to have C linkage / not be mangled
extern "C" { 
    void ISPCLaunch(void *f, void *data);
    void ISPCSync();
}


static int nThreads;
static pthread_t *threads;
static pthread_mutex_t taskQueueMutex;
static std::vector<std::pair<void *, void *> > taskQueue;
static sem_t *workerSemaphore;
static uint32_t numUnfinishedTasks;
static pthread_mutex_t tasksRunningConditionMutex;
static pthread_cond_t tasksRunningCondition;

static void *lTaskEntry(void *arg);

/** Figure out how many CPU cores there are in the system
 */
static int
lNumCPUCores() {
#if defined(__linux__)
    return sysconf(_SC_NPROCESSORS_ONLN);
#else
    // Mac
    int mib[2];
    mib[0] = CTL_HW;
    size_t length = 2;
    if (sysctlnametomib("hw.logicalcpu", mib, &length) == -1) {
        fprintf(stderr, "sysctlnametomib() filed.  Guessing 2 cores.");
        return 2;
    }
    assert(length == 2);

    int nCores = 0;
    size_t size = sizeof(nCores);

    if (sysctl(mib, 2, &nCores, &size, NULL, 0) == -1) {
        fprintf(stderr, "sysctl() to find number of cores present failed.  Guessing 2.");
        return 2;
    }
    return nCores;
#endif
}

void
TasksInit() {
    nThreads = lNumCPUCores();

    threads = new pthread_t[nThreads];

    int err;
    if ((err = pthread_mutex_init(&taskQueueMutex, NULL)) != 0) {
        fprintf(stderr, "Error creating mutex: %s\n", strerror(err));
        exit(1);
    }

    char name[32];
    sprintf(name, "mandelbrot.%d", (int)getpid());
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
        err = pthread_create(&threads[i], NULL, &lTaskEntry, reinterpret_cast<void *>(i));
        if (err != 0) {
            fprintf(stderr, "Error creating pthread %d: %s\n", i, strerror(err));
            exit(1);
        }
    }
}


void
ISPCLaunch(void *f, void *d) {
    //
    // Acquire mutex, add task
    //
    int err;
    if ((err = pthread_mutex_lock(&taskQueueMutex)) != 0) {
        fprintf(stderr, "Error from pthread_mutex_lock: %s\n", strerror(err));
        exit(1);
    }

    taskQueue.push_back(std::make_pair(f, d));

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
    int threadIndex = int(reinterpret_cast<int64_t>(arg));
    int threadCount = nThreads;

    while (true) {
        int err;
        if ((err = sem_wait(workerSemaphore)) != 0) {
            fprintf(stderr, "Error from sem_wait: %s\n", strerror(err));
            exit(1);
        }

        std::pair<void *, void *> myTask;
        //
        // Acquire mutex, get task
        //
        if ((err = pthread_mutex_lock(&taskQueueMutex)) != 0) {
            fprintf(stderr, "Error from pthread_mutex_lock: %s\n", strerror(err));
            exit(1);
        }
        if (taskQueue.size() == 0) {
            //
            // Task queue is empty, go back and wait on the semaphore
            //
            if ((err = pthread_mutex_unlock(&taskQueueMutex)) != 0) {
                fprintf(stderr, "Error from pthread_mutex_unlock: %s\n", strerror(err));
                exit(1);
            }
            continue;
        }

        myTask = taskQueue.back();
        taskQueue.pop_back();

        if ((err = pthread_mutex_unlock(&taskQueueMutex)) != 0) {
            fprintf(stderr, "Error from pthread_mutex_unlock: %s\n", strerror(err));
            exit(1);
        }

        //
        // Do work for _myTask_
        //
        typedef void (*TaskFunType)(void *, int, int);
        TaskFunType func = (TaskFunType)myTask.first;
        func(myTask.second, threadIndex, threadCount);

        //
        // Decrement the number of unfinished tasks counter
        //
        if ((err = pthread_mutex_lock(&tasksRunningConditionMutex)) != 0) {
            fprintf(stderr, "Error from pthread_mutex_lock: %s\n", strerror(err));
            exit(1);
        }

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
