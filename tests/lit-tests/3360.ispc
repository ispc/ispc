// RUN: not %{ispc} %s --nowrap --target=neon-i32x4 --emit-llvm-text --enable-llvm-intrinsics -o - 2>&1 | FileCheck %s

// CHECK: Error: LLVM intrinsic "ssse3.pmul.hr.sw.128" supported only on "x86" target architecture.
// CHECK-NOT: Assertion failed

// REQUIRES: ARM_ENABLED

int16 mul_16x8_high(int16 a, int16 b) {
    return @llvm.x86.ssse3.pmul.hr.sw.128(a, b);
}
