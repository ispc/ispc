// RUN: not %{ispc} %s --target=sse4.2-i16x8 --nowrap -o - 2>&1 | FileCheck %s

// CHECK: Error: Initializer list for varying "const varying int32" must have no more than 8 elements (has 16).
// CHECK-NOT: FATAL ERROR

// REQUIRES: X86_ENABLED
static const varying int shuffle_color_mask = { 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 };
