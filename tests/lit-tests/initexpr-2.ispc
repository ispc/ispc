// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Initializer list for array "varying int32[2][4]" must have no more than 2 elements (has 3)

int a[2][4] = { { 1, 2, 3 }, { 1, 2, 3, 4 }, 1 };
