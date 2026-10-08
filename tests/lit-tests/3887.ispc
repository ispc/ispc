// Regression test for #3887: the C builtins for 32-bit ARM (builtins_cpp_32_linux_armv8a.bc,
// the file that failed in the issue) and 64-bit ARM (builtins_cpp_64_linux_aarch64.bc) must
// be built and non-empty. On hosts without matching headers the build used to silently produce
// empty modules, leaving __do_print and __num_cores undefined.

// RUN: %{ispc} %s --arch=arm --target=neon-i32x4 --target-os=linux --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --target-os=linux --nowrap --emit-llvm-text -o - | FileCheck %s

// REQUIRES: ARM_ENABLED, LINUX_ENABLED

// CHECK-DAG: define {{.*}}@__do_print(
// CHECK-DAG: define {{.*}}@__num_cores(

void foo(uniform int out[]) {
    print("programIndex=%\n", programIndex);
    out[0] = num_cores();
}
