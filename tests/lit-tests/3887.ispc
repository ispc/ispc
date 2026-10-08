// Regression test for #3887: AArch64 C builtins must be non-empty, including when
// ARM32 is disabled. Incorrect header lookup used to silently produce empty modules.

// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --target-os=linux --nowrap --emit-llvm-text -o - | FileCheck %s

// REQUIRES: ARM_ENABLED, LINUX_ENABLED

// CHECK-DAG: define {{.*}}@__do_print(
// CHECK-DAG: define {{.*}}@__num_cores(

void foo(uniform int out[]) {
    print("programIndex=%\n", programIndex);
    out[0] = num_cores();
}
