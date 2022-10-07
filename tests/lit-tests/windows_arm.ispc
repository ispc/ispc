// RUN: %{ispc} %s --arch=aarch64  --target-os=windows --target=neon-i32x4 --emit-asm -o - | FileCheck %s --check-prefix=CHECK_ARM

// RUN: not %{ispc} %s --arch=arm --target-os=windows --target=neon-i32x4 --nowrap -o %t.o 2>&1 | FileCheck %s --check-prefix=CHECK_ERROR

// REQUIRES: WINDOWS_ENABLED && ARM_ENABLED

// CHECK_ERROR: Error: arm (32 bit) is not supported on Windows, use aarch64 instead.
float foo(float a, float b) {
// CHECK_ARM: fadd v0.4s
    return a + b;
}
