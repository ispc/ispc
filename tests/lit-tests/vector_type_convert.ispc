// This test checks that uniform vectors convertion is perfomed using vector instructions.
// RUN: %{ispc} %s --target=host --emit-llvm-text -o - | FileCheck %s
// CHECK: fpext <4 x float>
// CHECK-NOT: fpext float
void test(uniform float RET[], uniform float b) {
    uniform float<3> x = {1,2,3};
    uniform double<3> z = x+b;
    RET[programIndex] = z[programIndex];
}
