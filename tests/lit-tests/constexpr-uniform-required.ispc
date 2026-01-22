// REQUIRES: X86_ENABLED && X86_64_HOST
// RUN: not %{ispc} %s --target=generic-i32x4 --nowrap -o /dev/null 2>&1 | FileCheck %s

constexpr varying int Lane = programIndex;
uniform int arr[Lane];

// CHECK: Error: Array size must be a uniform compile-time constant.
