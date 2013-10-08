/*
  Copyright (c) 2011-2012, Intel Corporation
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

/*
  This file implements simple task systems that provide the three
  entrypoints used by ispc-generated to code to handle 'launch' and 'sync'
  statements in ispc programs.  See the section "Task Parallelism: Language
  Syntax" in the ispc documentation for information about using task
  parallelism in ispc programs, and see the section "Task Parallelism:
  Runtime Requirements" for information about the task-related entrypoints
  that are implemented here.

  There are several task systems in this file, built using:
    - Microsoft's Concurrency Runtime (ISPC_USE_CONCRT)
    - Apple's Grand Central Dispatch (ISPC_USE_GCD)
    - bare pthreads (ISPC_USE_PTHREADS, ISPC_USE_PTHREADS_FULLY_SUBSCRIBED)
    - Cilk Plus (ISPC_USE_CILK)
    - TBB (ISPC_USE_TBB_TASK_GROUP, ISPC_USE_TBB_PARALLEL_FOR)
    - OpenMP (ISPC_USE_OMP)

  The task system implementation can be selected at compile time, by defining 
  the appropriate preprocessor symbol on the command line (for e.g.: -D ISPC_USE_TBB).
  Not all combinations of platform and task system are meaningful.
  If no task system is requested, a reasonable default task system for the platform
  is selected.  Here are the task systems that can be selected:

#define ISPC_USE_GCD
#define ISPC_USE_CONCRT
#define ISPC_USE_PTHREADS
#define ISPC_USE_PTHREADS_FULLY_SUBSCRIBED
#define ISPC_USE_CILK
#define ISPC_USE_OMP
#define ISPC_USE_TBB_TASK_GROUP
#define ISPC_USE_TBB_PARALLEL_FOR

  The ISPC_USE_PTHREADS_FULLY_SUBSCRIBED model essentially takes over the machine
  by assigning one pthread to each hyper-thread, and then uses spinlocks and atomics
  for task management.  This model is useful for KNC where tasks can take over 
  the machine, but less so when there are other tasks that need running on the machine.

#define ISPC_USE_CREW

*/

#if !(defined ISPC_USE_CONCRT          || defined ISPC_USE_GCD              || \
      defined ISPC_USE_PTHREADS        || defined ISPC_USE_PTHREADS_FULLY_SUBSCRIBED || \
      defined ISPC_USE_TBB_TASK_GROUP  || defined ISPC_USE_TBB_PARALLEL_FOR || \
      defined ISPC_USE_OMP             || defined ISPC_USE_CILK             )

    // If no task model chosen from the compiler cmdline, pick a reasonable default
    #if defined(_WIN32) || defined(_WIN64)
      #define ISPC_USE_CONCRT
    #elif defined(__linux__)
    #define ISPC_USE_PTHREADS
    #elif defined(__APPLE__)
      #define ISPC_USE_GCD
    #endif
    #if defined(__KNC__)
      #define ISPC_USE_PTHREADS
    #endif

#endif // No task model specified on compiler cmdline

#if defined(_WIN32) || defined(_WIN64)
#define ISPC_IS_WINDOWS
#elif defined(__linux__)
#define ISPC_IS_LINUX
#elif defined(__APPLE__)
#define ISPC_IS_APPLE
#endif
#if defined(__KNC__)
#define ISPC_IS_KNC
#endif


#define DBG(x) 

#ifdef ISPC_IS_WINDOWS
  #define NOMINMAX
  #include <windows.h>
#endif // ISPC_IS_WINDOWS
#ifdef ISPC_USE_CONCRT
  #include <concrt.h>
  using namespace Concurrency;
#endif // ISPC_USE_CONCRT
#ifdef ISPC_USE_GCD
  #include <dispatch/dispatch.h>
  #include <pthread.h>
#endif // ISPC_USE_GCD
#ifdef ISPC_USE_PTHREADS
  #include <pthread.h>
  #include <semaphore.h>
  #include <unistd.h>
  #include <fcntl.h>
  #include <errno.h>
  #include <sys/types.h>
  #include <sys/stat.h>
  #include <sys/param.h>
  #include <sys/sysctl.h>
  #include <vector>
  #include <algorithm>
#endif // ISPC_USE_PTHREADS
#ifdef ISPC_USE_PTHREADS_FULLY_SUBSCRIBED
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/param.h>
#include <sys/sysctl.h>
#include <vector>
#include <algorithm>
//#include <stdexcept>
#include <stack>
#endif // ISPC_USE_PTHREADS_FULLY_SUBSCRIBED
#ifdef ISPC_USE_TBB_PARALLEL_FOR
  #include <tbb/parallel_for.h>
#endif // ISPC_USE_TBB_PARALLEL_FOR
#ifdef ISPC_USE_TBB_TASK_GROUP
  #include <tbb/task_group.h>
#endif // ISPC_USE_TBB_TASK_GROUP
#ifdef ISPC_USE_CILK
  #include <cilk/cilk.h>
#endif // ISPC_USE_TBB
#ifdef ISPC_USE_OMP
  #include <omp.h>
#endif // ISPC_USE_OMP
#ifdef ISPC_IS_LINUX
  #include <malloc.h>
#endif // ISPC_IS_LINUX

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <algorithm>

// Signature of ispc-generated 'task' functions
typedef void (*TaskFuncType)(void *data, int threadIndex, int threadCount,
                             int taskIndex, int taskCount);

// Small structure used to hold the data for each task
struct TaskInfo {
    TaskFuncType func;
    void *data;
    int taskIndex, taskCount;
#if defined(ISPC_IS_WINDOWS)
    event taskEvent;
#endif
};

// ispc expects these functions to have C linkage / not be mangled
extern "C" { 
    void ISPCLaunch(void **handlePtr, void *f, void *data, int count);
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

#ifdef ISPC_USE_CONCRT
// With ConcRT, we don't need to extend TaskGroupBase at all.
class TaskGroup : public TaskGroupBase {
public:
    void Launch(int baseIndex, int count);
    void Sync();
};
#endif // ISPC_USE_CONCRT

#ifdef ISPC_USE_GCD
/* With Grand Central Dispatch, we associate a GCD dispatch group with each
   task group.  (We'll later wait on this dispatch group when we need to
   wait on all of the tasks in the group to finish.)
 */
class TaskGroup : public TaskGroupBase {
public:
    TaskGroup() {
        gcdGroup = dispatch_group_create();
    }

    void Launch(int baseIndex, int count);
    void Sync();

private:
    dispatch_group_t gcdGroup;
};
#endif // ISPC_USE_GCD

#ifdef ISPC_USE_PTHREADS
static void *lTaskEntry(void *arg);

class TaskGroup : public TaskGroupBase {
public:
    TaskGroup() {
        numUnfinishedTasks = 0;
        waitingTasks.reserve(128);
        inActiveList = false;
    }

    void Reset() {
        TaskGroupBase::Reset();
        numUnfinishedTasks = 0;
        assert(inActiveList == false);
        lMemFence();
    }

    void Launch(int baseIndex, int count);
    void Sync();

private:
    friend void *lTaskEntry(void *arg);

    int32_t numUnfinishedTasks;
    int32_t pad[3];
    std::vector<int> waitingTasks;
    bool inActiveList;
};

#endif // ISPC_USE_PTHREADS

#ifdef ISPC_USE_CILK

class TaskGroup : public TaskGroupBase {
public:
    void Launch(int baseIndex, int count);
    void Sync();

};

#endif // ISPC_USE_CILK

#ifdef ISPC_USE_OMP

class TaskGroup : public TaskGroupBase {
public:
    void Launch(int baseIndex, int count);
    void Sync();

};

#endif // ISPC_USE_OMP

#ifdef ISPC_USE_TBB_PARALLEL_FOR

class TaskGroup : public TaskGroupBase {
public:
    void Launch(int baseIndex, int count);
    void Sync();

};

#endif // ISPC_USE_TBB_PARALLEL_FOR

#ifdef ISPC_USE_TBB_TASK_GROUP

class TaskGroup : public TaskGroupBase {
public:
    void Launch(int baseIndex, int count);
    void Sync();
private:
    tbb::task_group tbbTaskGroup;
};

#endif // ISPC_USE_TBB_TASK_GROUP

///////////////////////////////////////////////////////////////////////////
// Grand Central Dispatch

#ifdef ISPC_USE_GCD

/* A simple task system for ispc programs based on Apple's Grand Central
   Dispatch. */

static dispatch_queue_t gcdQueue;
static volatile int32_t lock = 0;

static void
InitTaskSystem() {
    if (gcdQueue != NULL)
        return;

    while (1) {
        if (lAtomicCompareAndSwap32(&lock, 1, 0) == 0) {
            if (gcdQueue == NULL) {
                gcdQueue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
                assert(gcdQueue != NULL);
                lMemFence();
            }
            lock = 0;
            break;
        }
    }
}


static void
lRunTask(void *ti) {
    TaskInfo *taskInfo = (TaskInfo *)ti;
    // FIXME: these are bogus values; may cause bugs in code that depends
    // on them having unique values in different threads.
    int threadIndex = 0;
    int threadCount = 1;

    // Actually run the task
    taskInfo->func(taskInfo->data, threadIndex, threadCount, 
                   taskInfo->taskIndex, taskInfo->taskCount);
}


inline void
TaskGroup::Launch(int baseIndex, int count) {
    for (int i = 0; i < count; ++i) {
        TaskInfo *ti = GetTaskInfo(baseIndex + i);
        dispatch_group_async_f(gcdGroup, gcdQueue, ti, lRunTask);
    }
}


inline void
TaskGroup::Sync() {
    dispatch_group_wait(gcdGroup, DISPATCH_TIME_FOREVER);
}

#endif // ISPC_USE_GCD

///////////////////////////////////////////////////////////////////////////
// Concurrency Runtime

#ifdef ISPC_USE_CONCRT

static void
InitTaskSystem() {
    // No initialization needed
}


static void __cdecl
lRunTask(LPVOID param) {
    TaskInfo *ti = (TaskInfo *)param;
    
    // Actually run the task. 
    // FIXME: like the GCD implementation for OS X, this is passing bogus
    // values for the threadIndex and threadCount builtins, which in turn
    // will cause bugs in code that uses those.
    int threadIndex = 0;
    int threadCount = 1;
    ti->func(ti->data, threadIndex, threadCount, ti->taskIndex, ti->taskCount);

    // Signal the event that this task is done
    ti->taskEvent.set();
}


inline void
TaskGroup::Launch(int baseIndex, int count) {
    for (int i = 0; i < count; ++i)
        CurrentScheduler::ScheduleTask(lRunTask, GetTaskInfo(baseIndex + i));
}


inline void
TaskGroup::Sync() {
    for (int i = 0; i < nextTaskInfoIndex; ++i) {
        TaskInfo *ti = GetTaskInfo(i);
        ti->taskEvent.wait();
        ti->taskEvent.reset();
    }
}

#endif // ISPC_USE_CONCRT

///////////////////////////////////////////////////////////////////////////
// pthreads

#ifdef ISPC_USE_PTHREADS

static volatile int32_t lock = 0;

static int nThreads;
static pthread_t *threads = NULL;

static pthread_mutex_t taskSysMutex;
static std::vector<TaskGroup *> activeTaskGroups;
static sem_t *workerSemaphore;

static void *
lTaskEntry(void *arg) {
    int threadIndex = (int)((int64_t)arg);
    int threadCount = nThreads;

    while (1) {
        int err;
        //
        // Wait on the semaphore until we're woken up due to the arrival of
        // more work.
        //
        if ((err = sem_wait(workerSemaphore)) != 0) {
            fprintf(stderr, "Error from sem_wait: %s\n", strerror(err));
            exit(1);
        }

        //
        // Acquire the mutex
        //
        if ((err = pthread_mutex_lock(&taskSysMutex)) != 0) {
            fprintf(stderr, "Error from pthread_mutex_lock: %s\n", strerror(err));
            exit(1);
        }

        if (activeTaskGroups.size() == 0) {
            //
            // Task queue is empty, go back and wait on the semaphore
            //
            if ((err = pthread_mutex_unlock(&taskSysMutex)) != 0) {
                fprintf(stderr, "Error from pthread_mutex_unlock: %s\n", strerror(err));
                exit(1);
            }
            continue;
        }

        //
        // Get the last task group on the active list and the last task
        // from its waiting tasks list.
        //
        TaskGroup *tg = activeTaskGroups.back();
        assert(tg->waitingTasks.size() > 0);
        int taskNumber = tg->waitingTasks.back();
        tg->waitingTasks.pop_back();

        if (tg->waitingTasks.size() == 0) {
            // We just took the last task from this task group, so remove
            // it from the active list.
            activeTaskGroups.pop_back();
            tg->inActiveList = false;
        }
    
        if ((err = pthread_mutex_unlock(&taskSysMutex)) != 0) {
            fprintf(stderr, "Error from pthread_mutex_unlock: %s\n", strerror(err));
            exit(1);
        }

        //
        // And now actually run the task
        //
        DBG(fprintf(stderr, "running task %d from group %p\n", taskNumber, tg));
        TaskInfo *myTask = tg->GetTaskInfo(taskNumber);
        myTask->func(myTask->data, threadIndex, threadCount, myTask->taskIndex,
                     myTask->taskCount);

        //
        // Decrement the "number of unfinished tasks" counter in the task
        // group.
        //
        lMemFence();
        lAtomicAdd(&tg->numUnfinishedTasks, -1);
    }

    pthread_exit(NULL);
    return 0;
}


static void
InitTaskSystem() {
    if (threads == NULL) {
        while (1) {
            if (lAtomicCompareAndSwap32(&lock, 1, 0) == 0) {
                if (threads == NULL) {
                    // We launch one fewer thread than there are cores,
                    // since the main thread here will also grab jobs from
                    // the task queue itself.
                    nThreads = sysconf(_SC_NPROCESSORS_ONLN) - 1;

                    int err;
                    if ((err = pthread_mutex_init(&taskSysMutex, NULL)) != 0) {
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

                    threads = (pthread_t *)malloc(nThreads * sizeof(pthread_t));
                    for (int i = 0; i < nThreads; ++i) {
                        err = pthread_create(&threads[i], NULL, &lTaskEntry, (void *)(i));
                        if (err != 0) {
                            fprintf(stderr, "Error creating pthread %d: %s\n", i, strerror(err));
                            exit(1);
                        }
                    }

                    activeTaskGroups.reserve(64);
                }

                // Make sure all of the above goes to memory before we
                // clear the lock.
                lMemFence();
                lock = 0;
                break;
            }
        }
    }
}


inline void
TaskGroup::Launch(int baseCoord, int count) {
    //
    // Acquire mutex, add task
    //
    int err;
    if ((err = pthread_mutex_lock(&taskSysMutex)) != 0) {
        fprintf(stderr, "Error from pthread_mutex_lock: %s\n", strerror(err));
        exit(1);
    }

    // Add the corresponding set of tasks to the waiting-to-be-run list for
    // this task group.
    //
    // FIXME: it's a little ugly to hold a global mutex for this when we
    // only need to make sure no one else is accessing this task group's
    // waitingTasks list.  (But a small experiment in switching to a
    // per-TaskGroup mutex showed worse performance!)
    for (int i = 0; i < count; ++i)
        waitingTasks.push_back(baseCoord + i);

    // Add the task group to the global active list if it isn't there
    // already.
    if (inActiveList == false) {
        activeTaskGroups.push_back(this);
        inActiveList = true;
    }

    if ((err = pthread_mutex_unlock(&taskSysMutex)) != 0) {
        fprintf(stderr, "Error from pthread_mutex_unlock: %s\n", strerror(err));
        exit(1);
    }

    //
    // Update the count of the number of tasks left to run in this task
    // group.
    //
    lMemFence();
    lAtomicAdd(&numUnfinishedTasks, count);

    //
    // Post to the worker semaphore to wake up worker threads that are
    // sleeping waiting for tasks to show up
    //
    for (int i = 0; i < count; ++i)
        if ((err = sem_post(workerSemaphore)) != 0) {
            fprintf(stderr, "Error from sem_post: %s\n", strerror(err));
            exit(1);
        }
}


inline void
TaskGroup::Sync() {
    DBG(fprintf(stderr, "syncing %p - %d unfinished\n", tg, numUnfinishedTasks));

    while (numUnfinishedTasks > 0) {
        // All of the tasks in this group aren't finished yet.  We'll try
        // to help out here since we don't have anything else to do...

        DBG(fprintf(stderr, "while syncing %p - %d unfinished\n", tg, 
                    numUnfinishedTasks));

        //
        // Acquire the global task system mutex to grab a task to work on
        //
        int err;
        if ((err = pthread_mutex_lock(&taskSysMutex)) != 0) {
            fprintf(stderr, "Error from pthread_mutex_lock: %s\n", strerror(err));
            exit(1);
        }

        TaskInfo *myTask = NULL;
        TaskGroup *runtg = this;
        if (waitingTasks.size() > 0) {
            int taskNumber = waitingTasks.back();
            waitingTasks.pop_back();

            if (waitingTasks.size() == 0) {
                // There's nothing left to start running from this group,
                // so remove it from the active task list.
                activeTaskGroups.erase(std::find(activeTaskGroups.begin(),
                                                 activeTaskGroups.end(), this));
                inActiveList = false;
            }
            myTask = GetTaskInfo(taskNumber);
            DBG(fprintf(stderr, "running task %d from group %p in sync\n", taskNumber, tg));
        }
        else {
            // Other threads are already working on all of the tasks in
            // this group, so we can't help out by running one ourself.
            // We'll try to run one from another group to make ourselves
            // useful here.
            if (activeTaskGroups.size() == 0) {
                // No active task groups left--there's nothing for us to do.
                if ((err = pthread_mutex_unlock(&taskSysMutex)) != 0) {
                    fprintf(stderr, "Error from pthread_mutex_unlock: %s\n", strerror(err));
                    exit(1);
                }
                // FIXME: We basically end up busy-waiting here, which is
                // extra wasteful in a world with hyper-threading.  It would
                // be much better to put this thread to sleep on a
                // condition variable that was signaled when the last task
                // in this group was finished.
#ifndef ISPC_IS_KNC
                usleep(1);
#else
                _mm_delay_32(8);
#endif
                continue;
            }

            // Get a task to run from another task group.
            runtg = activeTaskGroups.back();
            assert(runtg->waitingTasks.size() > 0);

            int taskNumber = runtg->waitingTasks.back();
            runtg->waitingTasks.pop_back();
            if (runtg->waitingTasks.size() == 0) {
                // There's left to start running from this group, so remove
                // it from the active task list.
                activeTaskGroups.pop_back();
                runtg->inActiveList = false;
            }
            myTask = runtg->GetTaskInfo(taskNumber);
            DBG(fprintf(stderr, "running task %d from other group %p in sync\n", 
                        taskNumber, runtg));
        }

        if ((err = pthread_mutex_unlock(&taskSysMutex)) != 0) {
            fprintf(stderr, "Error from pthread_mutex_unlock: %s\n", strerror(err));
            exit(1);
        }
    
        //
        // Do work for _myTask_
        //
        // FIXME: bogus values for thread index/thread count here as well..
        myTask->func(myTask->data, 0, 1, myTask->taskIndex, myTask->taskCount);

        //
        // Decrement the number of unfinished tasks counter
        //
        lMemFence();
        lAtomicAdd(&runtg->numUnfinishedTasks, -1);
    }
    DBG(fprintf(stderr, "sync for %p done!n", tg));
}

#endif // ISPC_USE_PTHREADS

///////////////////////////////////////////////////////////////////////////
// Cilk Plus

#ifdef ISPC_USE_CILK

static void
InitTaskSystem() {
    // No initialization needed
}

inline void
TaskGroup::Launch(int baseIndex, int count) {
    cilk_for(int i = 0; i < count; i++) {
        TaskInfo *ti = GetTaskInfo(baseIndex + i);

        // Actually run the task. 
        // Cilk does not expose the task -> thread mapping so we pretend it's 1:1
        ti->func(ti->data, ti->taskIndex, ti->taskCount, ti->taskIndex, ti->taskCount);
    }
}

inline void
TaskGroup::Sync() {
}

#endif // ISPC_USE_CILK

///////////////////////////////////////////////////////////////////////////
// OpenMP

#ifdef ISPC_USE_OMP

static void
InitTaskSystem() {
        // No initialization needed
}

inline void
TaskGroup::Launch(int baseIndex, int count) {
#pragma omp parallel for
    for(int i = 0; i < count; i++) {
        TaskInfo *ti = GetTaskInfo(baseIndex + i);

        // Actually run the task. 
        int threadIndex = omp_get_thread_num();
        int threadCount = omp_get_num_threads();
        ti->func(ti->data, threadIndex, threadCount, ti->taskIndex, ti->taskCount);
    }
}

inline void
TaskGroup::Sync() {
}

#endif // ISPC_USE_OMP

///////////////////////////////////////////////////////////////////////////
// Thread Building Blocks

#ifdef ISPC_USE_TBB_PARALLEL_FOR

static void
InitTaskSystem() {
    // No initialization needed by default
    //tbb::task_scheduler_init();
}

inline void
TaskGroup::Launch(int baseIndex, int count) {
    tbb::parallel_for(0, count, [=](int i) {
        TaskInfo *ti = GetTaskInfo(baseIndex + i);

        // Actually run the task. 
        // TBB does not expose the task -> thread mapping so we pretend it's 1:1
        int threadIndex = ti->taskIndex;
        int threadCount = ti->taskCount;

        ti->func(ti->data, threadIndex, threadCount, ti->taskIndex, ti->taskCount);
    });
}

inline void
TaskGroup::Sync() {
}

#endif // ISPC_USE_TBB_PARALLEL_FOR

#ifdef ISPC_USE_TBB_TASK_GROUP

static void
InitTaskSystem() {
    // No initialization needed by default
    //tbb::task_scheduler_init();
}

inline void
TaskGroup::Launch(int baseIndex, int count) {
    for (int i = 0; i < count; i++) {
        tbbTaskGroup.run([=]() {
            TaskInfo *ti = GetTaskInfo(baseIndex + i);

            // TBB does not expose the task -> thread mapping so we pretend it's 1:1
            int threadIndex = ti->taskIndex;
            int threadCount = ti->taskCount;
            ti->func(ti->data, threadIndex, threadCount, ti->taskIndex, ti->taskCount);
        });
    }
}

inline void
TaskGroup::Sync() {
    tbbTaskGroup.wait();
}

#endif // ISPC_USE_TBB_TASK_GROUP

///////////////////////////////////////////////////////////////////////////

#ifndef ISPC_USE_PTHREADS_FULLY_SUBSCRIBED

#define MAX_FREE_TASK_GROUPS 64
static TaskGroup *freeTaskGroups[MAX_FREE_TASK_GROUPS];

static inline TaskGroup *
AllocTaskGroup() {
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
FreeTaskGroup(TaskGroup *tg) {
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

///////////////////////////////////////////////////////////////////////////

void
ISPCLaunch(void **taskGroupPtr, void *func, void *data, int count) {
    TaskGroup *taskGroup;
    if (*taskGroupPtr == NULL) {
        InitTaskSystem();
        taskGroup = AllocTaskGroup();
        *taskGroupPtr = taskGroup;
    }
    else
        taskGroup = (TaskGroup *)(*taskGroupPtr);

    int baseIndex = taskGroup->AllocTaskInfo(count);
    for (int i = 0; i < count; ++i) {
        TaskInfo *ti = taskGroup->GetTaskInfo(baseIndex+i);
        ti->func = (TaskFuncType)func;
        ti->data = data;
        ti->taskIndex = i;
        ti->taskCount = count;
    }
    taskGroup->Launch(baseIndex, count);
}


void
ISPCSync(void *h) {
    TaskGroup *taskGroup = (TaskGroup *)h;
    if (taskGroup != NULL) {
        taskGroup->Sync();
        FreeTaskGroup(taskGroup);
    }
}


void *
ISPCAlloc(void **taskGroupPtr, int64_t size, int32_t alignment) {
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

#else  // ISPC_USE_PTHREADS_FULLY_SUBSCRIBED

#define MAX_LIVE_TASKS 1024

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// Small structure used to hold the data for each task
struct Task {
public:
    TaskFuncType func;
    void *data;
    volatile int32_t taskIndex;
    int taskCount;

    volatile int numDone;
    int liveIndex; // index in live task queue

    inline int  noMoreWork() { return taskIndex >= taskCount; }
    /*! given thread is done working on this task --> decrease num locks */
    // inline void lock() { lAtomicAdd(&locks,1); }
    // inline void unlock() { lAtomicAdd(&locks,-1); }
    inline int  nextJob() { return lAtomicAdd(&taskIndex,1); }
    inline int  numJobs() { return taskCount; }
    inline void schedule(int idx) { taskIndex = 0; numDone = 0; liveIndex = idx; }
    inline void run(int idx, int threadIdx);
    inline void markOneDone() { lAtomicAdd(&numDone,1); }
    inline void wait()
    {
        while (!noMoreWork()) {
            int next = nextJob();
            if (next < numJobs()) run(next, 0);
        }
        while (numDone != taskCount) {
#ifndef ISPC_IS_KNC
            usleep(1);
#else
            _mm_delay_32(8);
#endif
        }
    }
};

///////////////////////////////////////////////////////////////////////////
class TaskSys {
    static int numThreadsRunning;
    struct LiveTask
    {
        volatile int locks; /*!< num locks on this task. gets
                                 initialized to NUM_THREADS+1, then counted
                                 down by every thread that sees this. this
                                 value is only valid when 'active' is set
                                 to true */
        volatile int active; /*! workers will spin on this until it
                                 becomes active */
        Task *task;

        inline void doneWithThis() { lAtomicAdd(&locks,-1); }
        LiveTask() : active(0), locks(-1) {}
    };

public:
    volatile int nextScheduleIndex; /*! next index in the task queue
                                        where we'll insert a live task */

    // inline int inc_begin() { int old = begin; begin = (begin+1)%MAX_TASKS; return old; }
    // inline int inc_end() { int old = end; end = (end+1)%MAX_TASKS; return old; }

    LiveTask taskQueue[MAX_LIVE_TASKS];
    std::stack<Task *> taskMem;

    static TaskSys *global;

    TaskSys() : nextScheduleIndex(0)
    {
        TaskSys::global = this;
        Task *mem = new Task[MAX_LIVE_TASKS]; //< could actually be more than _live_ tasks
        for (int i=0;i<MAX_LIVE_TASKS;i++) {
            taskMem.push(mem+i);
        }
        createThreads();
    }

    inline Task *allocOne()
    {
        pthread_mutex_lock(&mutex);
        if (taskMem.empty()) {
            fprintf(stderr, "Too many live tasks.  "
                    "Change the value of MAX_LIVE_TASKS and recompile.\n");
            exit(1);
        }
        Task *task = taskMem.top();
        taskMem.pop();
        pthread_mutex_unlock(&mutex);
        return task;
    }

    static inline void init()
    {
        if (global) return;
        pthread_mutex_lock(&mutex);
        if (global == NULL) global = new TaskSys;
        pthread_mutex_unlock(&mutex);
    }

    void createThreads();
    int nThreads;
    pthread_t *thread;

    void threadFct();

    inline void schedule(Task *t)
    {
        pthread_mutex_lock(&mutex);
        int liveIndex = nextScheduleIndex;
        nextScheduleIndex = (nextScheduleIndex+1)%MAX_LIVE_TASKS;
        if (taskQueue[liveIndex].active) {
            fprintf(stderr, "Out of task queue resources.  "
                    "Change the value of MAX_LIVE_TASKS and recompile.\n");
            exit(1);
        }
        taskQueue[liveIndex].task = t;
        t->schedule(liveIndex);
        taskQueue[liveIndex].locks = numThreadsRunning+1; // num _worker_ threads plus creator
        taskQueue[liveIndex].active = true;
        pthread_mutex_unlock(&mutex);
    }

    void sync(Task *task)
    {
        task->wait();
        int liveIndex = task->liveIndex;
        while (taskQueue[liveIndex].locks > 1) {
#ifndef ISPC_IS_KNC
            usleep(1);
#else
            _mm_delay_32(8);
#endif
        }
        _mm_free(task->data);
        pthread_mutex_lock(&mutex);
        taskMem.push(task); // recycle task index
        taskQueue[liveIndex].active = false;
        pthread_mutex_unlock(&mutex);
    }
};


void TaskSys::threadFct() 
{
    int myIndex = 0; //lAtomicAdd(&threadIdx,1);
    while (1) {
        while (!taskQueue[myIndex].active) {
#ifndef ISPC_IS_KNC
            usleep(4);
#else
            _mm_delay_32(32);
#endif
            continue;
        }

        Task *mine = taskQueue[myIndex].task;
        while (!mine->noMoreWork()) {
            int job = mine->nextJob();
            if (job >= mine->numJobs()) break;
            mine->run(job,myIndex);
        }
        taskQueue[myIndex].doneWithThis();
        myIndex = (myIndex+1)%MAX_LIVE_TASKS;
    }
}


inline void Task::run(int idx, int threadIdx) {
    (*this->func)(data,threadIdx,TaskSys::global->nThreads,idx,taskCount);
    markOneDone();
}


void *_threadFct(void *data) {
    ((TaskSys*)data)->threadFct();
    return NULL;
}


void TaskSys::createThreads() 
{
    init();
    int reserved = 4;
    int minid = 2;
    nThreads = sysconf(_SC_NPROCESSORS_ONLN) - reserved;

    thread = (pthread_t *)malloc(nThreads * sizeof(pthread_t));

    numThreadsRunning = 0;
    for (int i = 0; i < nThreads; ++i) {
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 2*1024 * 1024);

        int threadID = minid+i;
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(threadID,&cpuset);
        int ret = pthread_attr_setaffinity_np(&attr,sizeof(cpuset),&cpuset);

        int err = pthread_create(&thread[i], &attr, &_threadFct, this);
        ++numThreadsRunning;
        if (err != 0) {
            fprintf(stderr, "Error creating pthread %d: %s\n", i, strerror(err));
            exit(1);
        }
    }
}

TaskSys * TaskSys::global = NULL;
int TaskSys::numThreadsRunning = 0;

///////////////////////////////////////////////////////////////////////////

void ISPCLaunch(void **taskGroupPtr, void *func, void *data, int count) 
{
    Task *ti = *(Task**)taskGroupPtr;
    ti->func = (TaskFuncType)func;
    ti->data = data;
    ti->taskIndex = 0;
    ti->taskCount = count;
    TaskSys::global->schedule(ti);
}

void ISPCSync(void *h) 
{
    Task *task = (Task *)h; 
    assert(task);
    TaskSys::global->sync(task);
}

void *ISPCAlloc(void **taskGroupPtr, int64_t size, int32_t alignment) 
{
    TaskSys::init();
    Task *task = TaskSys::global->allocOne();
    *taskGroupPtr = task;
    task->data = _mm_malloc(size,alignment);
    return task->data;//*taskGroupPtr;
}

#endif // ISPC_USE_PTHREADS_FULLY_SUBSCRIBED
