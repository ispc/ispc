// This test checks that compiler is not crashing and producing
// error when varying index is used with short vectors
// Currently fails (https://github.com/ispc/ispc/issues/2376).

// RUN: %{ispc} %s --target=host --nostdlib 2>&1 | FileCheck %s --allow-empty
// Error when LLVM built with no assertions
// CHECK-NOT: FATAL ERROR

typedef int<3> int3;

export void test(uniform float ret[], uniform float b) {
    int3 v0 = { b, 2*b, 3*b };
    int3 v1 = { b, b/2, b/3 };
    ret[programIndex] = (v0+v1)[programIndex];
}
