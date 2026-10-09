// The ARM32 builtin that failed in #3887 must contain its runtime implementations
// when 32-bit ARM libraries are enabled.
// RUN: %{ispc} %s --arch=arm --target=neon-i32x4 --target-os=linux --nowrap --emit-llvm-text -o - | FileCheck %s
// REQUIRES: ARM_32BIT_ENABLED, LINUX_ENABLED

// CHECK-DAG: define {{.*}}@__do_print(
// CHECK-DAG: define {{.*}}@__num_cores(
void foo(uniform int out[]) {
    print("programIndex=%\n", programIndex);
    out[0] = num_cores();
}
