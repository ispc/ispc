// Verifies that the Windows C builtins are not empty, also when ispc is built on a
// non-Windows host with the Windows SDK headers: if clang fails on them,
// __do_print and __num_cores are only declared, and programs using print() and
// num_cores() fail to link.
// See https://github.com/ispc/ispc/issues/3887

// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --target-os=windows --nowrap --emit-llvm-text -o - | FileCheck %s

// REQUIRES: ARM_ENABLED && WINDOWS_ENABLED

// CHECK-DAG: define {{.*}} @__do_print(
// CHECK-DAG: define {{.*}} @__num_cores(

void foo(uniform int out[]) {
    print("programIndex=%\n", programIndex);
    out[0] = num_cores();
}
