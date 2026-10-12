// Verifies that building with ARM32_ENABLED=OFF removes 32-bit ARM support only:
// --arch=arm is rejected for both neon and generic targets, while aarch64 still
// compiles with its stdlib, so --nostdlib is not used.
// Also verifies that the aarch64 C builtins are not empty: if clang fails on the
// C library headers, __do_print and __num_cores are only declared, and programs
// using print() and num_cores() fail to link.
// See https://github.com/ispc/ispc/issues/3887

// RUN: not %{ispc} %s --arch=arm --target=neon-i32x4 --target-os=linux --nowrap -o %t.o 2>&1 | FileCheck %s --check-prefix=CHECK_ARM_NEON
// RUN: not %{ispc} %s --arch=arm --target=generic-i32x4 --target-os=linux --nowrap -o %t.o 2>&1 | FileCheck %s --check-prefix=CHECK_ARM_GENERIC
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --target-os=linux --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK_BUILTINS
// RUN: %{ispc} %s --arch=aarch64 --target=generic-i32x4 --target-os=linux --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK_BUILTINS

// REQUIRES: ARM_ENABLED && LINUX_ENABLED && !ARM32_ENABLED

// CHECK_ARM_NEON: Error: neon-i32x4 target for arm on Linux is not supported in current build.
// CHECK_ARM_GENERIC: Error: generic-i32x4 target for arm on Linux is not supported in current build.

// CHECK_BUILTINS-DAG: define {{.*}} @__do_print(
// CHECK_BUILTINS-DAG: define {{.*}} @__num_cores(

void foo(uniform int out[]) {
    print("programIndex=%\n", programIndex);
    out[0] = num_cores();
}
