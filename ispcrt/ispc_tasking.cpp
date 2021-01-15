// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <stdint.h>
#include <stdlib.h>

// Signature of ispc-generated 'task' functions
typedef void (*TaskFuncType)(void *data, int threadIndex, int threadCount, int taskIndex, int taskCount, int taskIndex0,
                             int taskIndex1, int taskIndex2, int taskCount0, int taskCount1, int taskCount2);

#ifdef _OPENMP
#include <omp.h>

extern "C" void ISPCLaunch(void **taskGroupPtr, void *_func, void *data, int count0, int count1, int count2) {
    const int count = count0 * count1 * count2;
    TaskFuncType func = (TaskFuncType)_func;

#pragma omp parallel
    {
        const int threadIndex = omp_get_thread_num();
        const int threadCount = omp_get_num_threads();

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
#else
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#if defined(_WIN32) || defined(_WIN64)
#include <malloc.h>
#endif

using TaskFcn = std::function<void(size_t)>;

struct Task {
    bool completed() const { return iterationsCompleted == totalIterations; }

    void computeNextIteration() {
        size_t taskID = iterationsStarted++;
        if (taskID < totalIterations) {
            fcn(taskID);
            iterationsCompleted++;
        }
    }

    // Data //

    size_t totalIterations{0};
    alignas(64) std::atomic<size_t> iterationsStarted{0};
    alignas(64) std::atomic<size_t> iterationsCompleted{0};
    TaskFcn fcn;
};

struct TaskSystem {
    TaskSystem(int numThreads = 0);
    ~TaskSystem();

    void schedule(size_t numIterations, TaskFcn fcn);

  private:
    Task *getCurrentTask();

    // Data //

    std::atomic_bool threadsRunning{false};
    std::atomic_int runningThreads{0};
    int numThreads{0};
    Task *currentTask{nullptr};
    std::mutex taskMutex;
    std::condition_variable tasksAvailable;
};

std::unique_ptr<TaskSystem> g_ts;

TaskSystem::TaskSystem(int _numThreads) {
    threadsRunning = true;

    if (_numThreads <= 0)
        _numThreads = std::max(std::thread::hardware_concurrency() - 1, 1u);

    numThreads = _numThreads;

    for (int i = 0; i < numThreads; i++) {
        std::thread thread([&]() {
            runningThreads++;

            while (threadsRunning) {
                Task *task = getCurrentTask();

                if (task == nullptr || task->completed())
                    continue;

                task->computeNextIteration();
            }

            runningThreads--;
        });

        thread.detach();
    }
}

TaskSystem::~TaskSystem() {
    threadsRunning = false;
    tasksAvailable.notify_all();
    while (runningThreads > 0)
        ;
}

void TaskSystem::schedule(size_t numIterations, TaskFcn fcn) {
    std::unique_ptr<Task> task{new Task()};

    task->totalIterations = numIterations;
    task->fcn = std::move(fcn);

    currentTask = task.get();

    tasksAvailable.notify_all();

    while (!task->completed())
        task->computeNextIteration();

    currentTask = nullptr;
}

Task *TaskSystem::getCurrentTask() {
    if (currentTask != nullptr)
        return currentTask;

    {
        std::unique_lock<std::mutex> lock(taskMutex);
        tasksAvailable.wait(lock, [&]() { return !(currentTask == nullptr && threadsRunning); });
        return currentTask;
    }
}

extern "C" void ISPCLaunch(void **taskGroupPtr, void *_func, void *data, int count0, int count1, int count2) {
    const size_t count = size_t(count0) * size_t(count1) * size_t(count2);
    TaskFuncType func = (TaskFuncType)_func;

    if (!g_ts.get())
        g_ts.reset(new TaskSystem());

    g_ts->schedule(count, [&](size_t i) {
        int taskIndex0 = i % count0;
        int taskIndex1 = (i / count0) % count1;
        int taskIndex2 = i / (count0 * count1);

        func(data, i, count, i, count, taskIndex0, taskIndex1, taskIndex2, count0, count1, count2);
    });
}
#endif

extern "C" void ISPCSync(void *h) { free(h); }

extern "C" void *ISPCAlloc(void **taskGroupPtr, int64_t size, int32_t alignment) {
#if defined(_WIN32) || defined(_WIN64)
    *taskGroupPtr = _aligned_malloc(size, alignment);
#else
    *taskGroupPtr = aligned_alloc(alignment, size);
#endif
    return *taskGroupPtr;
}
