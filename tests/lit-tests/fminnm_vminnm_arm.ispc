// Tests checks that IEEE 754 compliant instructions are generated for max/min operations on Arm.

// RUN: %{ispc} %s --arch=arm --target=neon-i32x4 --cpu=cortex-a35 --emit-asm  -o - | FileCheck %s
// RUN: %{ispc} %s --arch=arm --target=neon-i32x8 --cpu=cortex-a35 --emit-asm  -o - | FileCheck %s
// RUN: %{ispc} %s --arch=arm --target=neon-i32x4 --target-os=custom_linux --emit-asm  -o - | FileCheck %s
// RUN: %{ispc} %s --arch=arm --target=neon-i32x8 --target-os=custom_linux --emit-asm  -o - | FileCheck %s

// ARM must be enabled in order to test it.
// Windows and Mac do not support 32 bit ARM. so test on Linux only (Android assumed to be the same)
// REQUIRES: ARM_ENABLED && LINUX_HOST

// CHECK-LABEL: test_max___vyfvyf:
// CHECK: vmaxnm.f32
float test_max(float a, float b){
    return max(a, b);
}

// CHECK-LABEL: test_min___vyfvyf:
// CHECK: vminnm.f32
float test_min(float a, float b){
    return min(a, b);
}

