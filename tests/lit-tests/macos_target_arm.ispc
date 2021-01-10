// RUN: %{ispc} %s --arch=aarch64  --target-os=macos --target=neon-i32x4 --emit-asm -o - | FileCheck %s --check-prefix=CHECK_ARM

// RUN: not %{ispc} %s --arch=arm --target-os=macos --target=neon-i32x4 --nowrap -o %t.o 2>&1 | FileCheck %s --check-prefix=CHECK_ERROR

// REQUIRES: MACOS_ARM_ENABLED

// CHECK_ERROR: Error: macOS target supports only x86_64 and aarch64.
float foo(float a, float b) {
// CHECK_ARM: fadd.4s
    return a + b;
}
