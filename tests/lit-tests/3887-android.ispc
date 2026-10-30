// Verifies that the Android C builtins are not empty: if clang fails on them,
// __do_print and __num_cores are only declared, and programs using print() and
// num_cores() fail to link. Also verifies that they use the bionic stdout symbol
// and _SC_NPROCESSORS_ONLN value.
// See https://github.com/ispc/ispc/issues/3887

// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --target-os=android --nowrap --emit-llvm-text -o - | FileCheck %s

// REQUIRES: ARM_ENABLED && ANDROID_ENABLED

// CHECK-DAG: define {{.*}} @__do_print(
// CHECK-DAG: define {{.*}} @__num_cores(
// CHECK-DAG: @stdout = external
// CHECK-DAG: call {{.*}} @sysconf(i32 {{.*}}97)

void foo(uniform int out[]) {
    print("programIndex=%\n", programIndex);
    out[0] = num_cores();
}
