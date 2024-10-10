// Tests checks that IEEE 754 compliant instructions are generated for max/min operations on AARCH64.

// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --emit-asm  -o - | FileCheck %s
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x8 --emit-asm  -o - | FileCheck %s

// REQUIRES: ARM_ENABLED

// CHECK-LABEL: test_max___vyfvyf:
// CHECK: fmaxnm
float test_max(float a, float b){
    return max(a, b);
}
// CHECK-LABEL: test_min___vyfvyf:
// CHECK: fminnm
float test_min(float a, float b){
    return min(a, b);
}
