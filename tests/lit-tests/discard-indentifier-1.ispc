// RUN: not %{ispc} --target=host --nostdlib --nowrap -o - %s 2>&1 | FileCheck %s

// CHECK: Error: syntax error
// CHECK-NOT: FATAL ERROR: Unhandled signal sent to process


const int r = 7;

r = 1;
