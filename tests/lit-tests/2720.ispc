// The tests checks that no extra load/store instructions is generated.
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --emit-asm  -o - | FileCheck %s
// REQUIRES: ARM_ENABLED

// CHECK-COUNT: 1 mov
// CHECK-COUNT: 1 str
// CHECK-COUNT: 3 ldr
// CHECK-COUNT: 3 ldr
// CHECK-NOT: add
uniform int foo(uniform float v1[], uniform float v2[], uniform float v3[], uniform float result[]) {
    uniform float invArea = 1.f / ((v3[0] - v1[0]) * (v2[1] - v1[1]) - ((v3[1] - v1[1]) * (v2[0] - v1[0])));
    result[0] = invArea;
    result[1] = invArea;
    result[2] = invArea;
    result[3] = invArea;
}
