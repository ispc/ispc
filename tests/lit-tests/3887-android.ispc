// Android C builtins must contain their runtime implementations. Host C++
// library headers and unversioned Android triples used to prevent generation.
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --target-os=android --nowrap --emit-llvm-text -o - | FileCheck %s
// REQUIRES: ARM_ENABLED, ANDROID_ENABLED

// CHECK-DAG: define {{.*}}@__do_print(
// CHECK-DAG: define {{.*}}@__num_cores(
void foo(uniform int out[]) {
    print("programIndex=%\n", programIndex);
    out[0] = num_cores();
}
