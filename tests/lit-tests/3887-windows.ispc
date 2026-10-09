// Windows C builtins must contain their runtime implementations when generated
// on other hosts as well as on Windows.
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --target-os=windows --nowrap --emit-llvm-text -o - | FileCheck %s
// REQUIRES: ARM_ENABLED, WINDOWS_ENABLED

// CHECK-DAG: define {{.*}}@__do_print(
// CHECK-DAG: define {{.*}}@__num_cores(
void foo(uniform int out[]) {
    print("programIndex=%\n", programIndex);
    out[0] = num_cores();
}
