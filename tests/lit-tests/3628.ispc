// Test for issue #3628: Warning when calling exported function from ISPC code
// This warning prepares users for the future default behavior change where
// exported functions will only generate external versions by default.

// RUN: %{ispc} %s --target=host --nostdlib --nowrap -o %t.o 2>&1 | FileCheck %s

extern void foo();
export void exported_func() {
    // Empty function for testing
    foo();
}

// Regular non-exported function should not generate warning
void regular_func() {
    // Empty function for testing
    foo();
}

void caller() {
    // CHECK: [[# @LINE + 1]]:5: Warning: Calling exported function from ISPC code
    exported_func();

    // CHECK-NOT: Warning
    regular_func();
}
