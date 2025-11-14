// Test that calling external_only and export functions when "--no-internal-export-functions" is used
// from ISPC code produces an error (both direct calls and through function pointers)
// RUN: not %{ispc} %s --target=host --nowrap --nostdlib -o %t.o 2>&1 | FileCheck %s
// RUN: not %{ispc} %s --target=host --nowrap --nostdlib --no-internal-export-functions -o %t.o 2>&1 | FileCheck %s --check-prefix=CHECK_FLAG

__attribute__((external_only)) export void external_only_func() {}

export void exported_func() {}

typedef export void (*uniform FuncPtr)();

void caller_direct() {
    // CHECK: [[# @LINE + 1]]:5: Error: Calling exported function with no internal version from ISPC code
    external_only_func();
}

void caller2_direct() {
    // CHECK_FLAG: [[# @LINE + 1]]:5: Error: Calling exported function with no internal version from ISPC code
    exported_func();
}

void caller2_via_pointer() {
    FuncPtr fptr = exported_func;
    // CHECK_FLAG: [[# @LINE + 1]]:5: Error: Calling exported function with no internal version from ISPC code
    fptr();
}
