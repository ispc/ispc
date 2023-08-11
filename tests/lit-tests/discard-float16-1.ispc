// RUN: not %{ispc} --target=host --nostdlib --nowrap -o - %s 2>&1 | FileCheck %s

// CHECK: Error: syntax error
// CHECK-NOT: FATAL ERROR: Unhandled signal sent to process

float16 2.0f16 = two_f16;
