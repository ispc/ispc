// Test for issue #3628: Warning when calling exported function from ISPC code
// This warning prepares users for the future default behavior change where
// exported functions will only generate external versions by default.
// Note: Error cases (external_only, --no-internal-export-functions) are tested in 3628-error.ispc

// RUN: %{ispc} %s --target=host --nostdlib --nowrap -o %t.o 2>&1 | FileCheck %s

extern void foo();
export void exported_func() {
    foo();
}

// Regular non-exported function should not generate warning
void regular_func() {
    foo();
}

void caller_direct() {
    // CHECK: [[# @LINE + 1]]:5: Warning: Calling exported function from ISPC code
    exported_func();

    // CHECK-NOT: Warning
    regular_func();
}

typedef export void (*uniform FuncPtrExport)();
typedef void (*uniform FuncPtr)();

void caller_via_pointer() {
    FuncPtrExport fptr = exported_func;
    // CHECK: [[# @LINE + 1]]:5: Warning: Calling exported function from ISPC code
    fptr();

    FuncPtr fptr2 = regular_func;
    // CHECK-NOT: Warning
    fptr2();
}
