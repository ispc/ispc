// RUN: not %{ispc} %s --target=generic-i32x4 -o /dev/null 2>&1 | FileCheck %s

constexpr varying int Lane = programIndex;
uniform int arr[Lane];

// CHECK: Error: Array size must
// CHECK: be a uniform compile-time constant.
