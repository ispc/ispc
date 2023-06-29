// Check that ISPC produces error for incorrect code and does not crash. ISPC issue #2523.
// RUN: not %{ispc} %s --nowrap --nostdlib --target=host 2>&1 | FileCheck %s

// CHECK: Error: Premature end of file: syntax error
// CHECK-NOT: FATAL ERROR: Unhandled signal sent to process
void foobar() { {
}
