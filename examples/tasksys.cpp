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

/*
  This file implements simple task systems that provide the three
  entrypoints used by ispc-generated to code to handle 'launch' and 'sync'
  statements in ispc programs.  See the section "Task Parallelism: Language
  Syntax" in the ispc documentation for information about using task
  parallelism in ispc programs, and see the section "Task Parallelism:
  Runtime Requirements" for information about the task-related entrypoints
  that are implemented here.

  There are three task systems in this file: one built using Microsoft's
  Concurrency Runtime, one built with Apple's Grand Central Dispatch, and
  one built on top of bare pthreads.
*/

#if defined(_WIN32) || defined(_WIN64)
  #define ISPC_IS_WINDOWS
  #define ISPC_USE_CONCRT
#elif defined(__linux__)
  #define ISPC_IS_LINUX
  #define ISPC_USE_PTHREADS
#elif defined(__APPLE__)
  #define ISPC_IS_APPLE
  #define ISPC_USE_GCD
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
        delete[] memBuffers[i];
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
    int64_t iptr = (int64_t)(basePtr + curMemBufferOffset);
    iptr = (iptr + (alignment-1)) & ~(alignment-1);

    int newOffset = int(iptr + size - (int64_t)basePtr);
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

#ifndef ISPC_IS_WINDOWS
static inline void
lMemFence() {
    __asm__ __volatile__("mfence":::"memory");
}
#endif // !ISPC_IS_WINDOWS


#if (__SIZEOF_POINTER__ == 4) || defined(__i386__) || defined(_WIN32)
#define ISPC_POINTER_BYTES 4
#elif (__SIZEOF_POINTER__ == 8) || defined(__x86_64__) || defined(__amd64__) || defined(_WIN64)
#define ISPC_POINTER_BYTES 8
#else
#error "Pointer size unknown!"
#endif // __SIZEOF_POINTER__


static void *
lAtomicCompareAndSwapPointer(void **v, void *newValue, void *oldValue) {
#ifdef ISPC_IS_WINDOWS
    return InterlockedCompareExchangePointer(v, newValue, oldValue);
#else
    void *result;
#if (ISPC_POINTER_BYTES == 4)
    __asm__ __volatile__("lock\ncmpxchgl %2,%1"
                          : "=a"(result), "=m"(*v)
                          : "q"(newValue), "0"(oldValue)
                          : "memory");
#else
    __asm__ __volatile__("lock\ncmpxchgq %2,%1"
                          : "=a"(result), "=m"(*v)
                          : "q"(newValue), "0"(oldValue)
                          : "memory");
#endif // ISPC_POINTER_BYTES
    lMemFence();
    return result;
#endif // ISPC_IS_WINDOWS
}



#ifndef ISPC_IS_WINDOWS
static int32_t 
lAtomicCompareAndSwap32(volatile int32_t *v, int32_t newValue, int32_t oldValue) {
    int32_t result;
    __asm__ __volatile__("lock\ncmpxchgl %2,%1"
                          : "=a"(result), "=m"(*v)
                          : "q"(newValue), "0"(oldValue)
                          : "memory");
    lMemFence();
    return result;
}
#endif // !ISPC_IS_WINDOWS


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


static inline int32_t 
lAtomicAdd(int32_t *v, int32_t delta) {
    int32_t origValue;
    __asm__ __volatile__("lock\n"
                         "xaddl %0,%1"
                         : "=r"(origValue), "=m"(*v) : "0"(delta)
                         : "memory");
    return origValue;
}


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
                // extra wasteful in a world with hyperthreading.  It would
                // be much better to put this thread to sleep on a
                // condition variable that was signaled when the last task
                // in this group was finished.
                sleep(0);
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

#define MAX_FREE_TASK_GROUPS 64
static TaskGroup *freeTaskGroups[MAX_FREE_TASK_GROUPS];

static inline TaskGroup *
AllocTaskGroup() {
    for (int i = 0; i < MAX_FREE_TASK_GROUPS; ++i) {
        TaskGroup *tg = freeTaskGroups[i];
        if (tg != NULL) {
            void *ptr = lAtomicCompareAndSwapPointer((void **)(&freeTaskGroups[i]), NULL, tg);
            if (ptr != NULL) {
                assert(ptr == tg);
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

// ispc expects these functions to have C linkage / not be mangled
extern "C" { 
    void ISPCLaunch(void **handlePtr, void *f, void *data, int count);
    void *ISPCAlloc(void **handlePtr, int64_t size, int32_t alignment);
    void ISPCSync(void *handle);
}

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
