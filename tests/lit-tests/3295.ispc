// This test checks that the compiler reports an error when a pointer type is
// used with unsupported binary operators.

// RUN: not %{ispc} --target=host --nowrap %s -o - 2>&1 | FileCheck %s

// CHECK-NOT: Error: Assertion failed
// CHECK: Error: First operand to binary operator "%" is of invalid type "uniform int32 * uniform".
// CHECK: Error: Unsupported binary operator "%" for pointer type: "uniform int32 * uniform"


export uniform int gather_ints_buffer(uniform int values[]) { assume(&values[0] % programCount == 0); }
