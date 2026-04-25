// Test for LLVM trunk regression where ConstantFP with vector type caused assertion failure
// in lVectorValuesAllEqual. This test verifies that scattering uniform float values compiles
// without crashing at -O0 (where the scatter pattern is most likely to trigger the code path).
//
// RUN: %{ispc} %s --target=host --emit-llvm-text -O0 -o - | FileCheck %s
// REQUIRES: X86_ENABLED

// CHECK: define void @scatter_uniform_float
void scatter_uniform_float(uniform float out[]) {
    uniform float x[8];
    for (uniform int i = 0; i < 8; ++i)
        x[i] = 0.0;

    out[programIndex] = x[programIndex];
}
