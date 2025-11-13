//; RUN: %{ispc} %s --target=host --nowrap -o %t.o 2>&1 | FileCheck %s

// Test for issue #3628: Warning when calling exported function from ISPC code
// This warning prepares users for the future default behavior change where
// exported functions will only generate external versions by default.

// CHECK: [[# @LINE + 1]]:5: Warning: Calling exported function from ISPC code
export void exported_func() {
    // Empty function for testing
}

// Function with external_only attribute should not generate warning
// CHECK-NOT: [[# @LINE + 2]]:5: Warning
__attribute__((external_only))
export void external_only_func() {
    // Empty function for testing
}

// Regular non-exported function should not generate warning
// CHECK-NOT: [[# @LINE + 1]]:5: Warning
void regular_func() {
    // Empty function for testing
}

void caller() {
    // This should trigger a warning because we're calling an exported function
    // from ISPC code
    exported_func();

    // This should NOT trigger a warning because external_only is explicitly set
    external_only_func();

    // This should NOT trigger a warning because it's not exported
    regular_func();
}
