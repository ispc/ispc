// The test checks compiler diagnostics when task-related primitives like
// (taskIndex, taskCount etc) are used in not standard ways:
// 1. as a function parameter
// 2. as a function name
// 3. as a local variable
// Compiler should produce a proper error and not crash. The diagnostics should
// be the same for CPU and Xe targets.

// RUN: not %{ispc} %s -o %t.o --nowrap -DEXPORT_FUNC 2>&1 | FileCheck %s -check-prefix=CHECK-EXPORT_EXTERN_FUNC
// RUN: not %{ispc} %s -o %t.o --target=gen9-x8 --nowrap -DEXPORT_FUNC 2>&1 | FileCheck %s -check-prefix=CHECK-EXPORT_EXTERN_FUNC

// RUN: not %{ispc} %s -o %t.o --nowrap -DTASK_FUNC 2>&1 | FileCheck %s -check-prefix=CHECK-TASK_FUNC
// RUN: not %{ispc} %s -o %t.o --target=gen9-x8 --nowrap -DTASK_FUNC 2>&1 | FileCheck %s -check-prefix=CHECK-TASK_FUNC

// RUN: not %{ispc} %s -o %t.o --nowrap 2>&1 | FileCheck %s -check-prefix=CHECK-EXPORT_EXTERN_FUNC
// RUN: not %{ispc} %s -o %t.o --target=gen9-x8 --nowrap 2>&1 | FileCheck %s -check-prefix=CHECK-EXPORT_EXTERN_FUNC

// CHECK-EXPORT_EXTERN_FUNC: Error: Undeclared symbol "taskIndex0".
// CHECK-EXPORT_EXTERN_FUNC: Error: Undeclared symbol "taskIndex1".
// CHECK-EXPORT_EXTERN_FUNC: Error: Undeclared symbol "taskIndex2".
// CHECK-EXPORT_EXTERN_FUNC: Error: Undeclared symbol "threadIndex".
// CHECK-EXPORT_EXTERN_FUNC: Error: Undeclared symbol "threadCount".
// CHECK-EXPORT_EXTERN_FUNC: Warning: Function parameter "taskIndex" shadows a function declared in global scope.
// CHECK-EXPORT_EXTERN_FUNC-NOT: FATAL ERROR

// CHECK-TASK_FUNC: Error: Ignoring redeclaration of symbol "taskIndex".
// CHECK-TASK_FUNC: Warning: Symbol "taskCount" shadows symbol declared in outer scope.
// CHECK-TASK_FUNC: Warning: Function parameter "taskIndex" shadows a function declared in global scope.
// CHECK-TASK_FUNC: Error: Ignoring redeclaration of symbol "taskIndex".
// CHECK-TASK_FUNC-NOT: FATAL ERROR

// REQUIRES: XE_ENABLED

#ifdef EXPORT_FUNC
export void export_fn1(const uniform int taskIndex) { }

export void export_fn2(uniform int res[]) {
    res[0] = taskIndex0 * taskIndex1 * taskIndex2;
    res[1] = threadIndex + threadCount;
}

export void export_fn3(uniform int res[]) {
    uniform int taskCount = programCount;
    res[0] = taskCount;
}

int taskIndex();
export void export_fn4(uniform int* uniform taskIndex) { }

#elif TASK_FUNC
task void task_fn1(const uniform int taskIndex) { }

task void task_fn2(uniform int res[]) {
    res[0] = taskIndex0 * taskIndex1 * taskIndex2;
    res[1] = threadIndex + threadCount;
}

task void task_fn3(uniform int res[]) {
    uniform int taskCount = programCount;
    res[0] = taskCount;
}

int taskIndex();
task void task_fn4(uniform int* uniform taskIndex) { }

#else
extern void extern_fn1(const uniform int taskIndex) { }

extern void extern_fn2(uniform int res[]) {
    res[0] = taskIndex0 * taskIndex1 * taskIndex2;
    res[1] = threadIndex + threadCount;
}

extern void extern_fn3(uniform int res[]) {
    uniform int taskCount = programCount;
    res[0] = taskCount;
}

int taskIndex();
extern void extern_fn4(uniform int* uniform taskIndex) { }
#endif