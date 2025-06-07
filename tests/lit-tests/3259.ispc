// This test checks that the compiler reports an error regarding the invalid 
// `&` in parameters and that this does not cause any crash (fatal error).

// RUN: not %{ispc} --target=host --nowrap %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Arrays of references (type "uniform float &") are illegal.
// CHECK-NOT: FATAL ERROR:

export void float3_dot_product(const uniform float values[][3],
                               const uniform float factors[],
                               const uniform int count,
                               float uniform &result[3]) // Caused by `&`
{
  foreach (index = 0 ... count, axis = 0 ... 3) {
    result[axis] += values[index][axis] * factors[index];
  }
}
