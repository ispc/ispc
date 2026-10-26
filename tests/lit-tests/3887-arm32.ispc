// Verifies that with ARM32_ENABLED=ON the 32-bit ARM and AArch64 Linux C builtins
// are not empty: if clang fails on the C library headers, __do_print and
// __num_cores are only declared, and programs using print() and num_cores() fail
// to link. 3887.ispc covers builds with ARM32_ENABLED=OFF.
// See https://github.com/ispc/ispc/issues/3887

// RUN: %{ispc} %s --arch=arm --target=neon-i32x4 --target-os=linux --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --target-os=linux --nowrap --emit-llvm-text -o - | FileCheck %s

// REQUIRES: ARM32_ENABLED && LINUX_ENABLED

// CHECK-DAG: define {{.*}} @__do_print(
// CHECK-DAG: define {{.*}} @__num_cores(

void foo(uniform int out[]) {
    print("programIndex=%\n", programIndex);
    out[0] = num_cores();
}
