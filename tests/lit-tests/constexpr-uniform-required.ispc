// RUN: not %{ispc} %s %{generic_i32x4} --nostdlib --nowrap -o /dev/null 2>&1 | FileCheck %s
// REQUIRES: GENERIC_I32X4

constexpr varying int Lane = programIndex;
uniform int arr[Lane];

// CHECK: Error: Array size must be a uniform compile-time constant.
