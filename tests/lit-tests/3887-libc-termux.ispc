// Verifies that the Linux C builtins of ISPC built on Android (Termux) use the
// bionic stdout symbol and _SC_NPROCESSORS_ONLN value: the C library there is
// bionic. The builtins are built without the C library headers (#3887).
// See https://github.com/ispc/ispc/issues/3887

// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --target-os=linux --nowrap --emit-llvm-text -o - | FileCheck %s

// REQUIRES: ANDROID_HOST && ARM_ENABLED && LINUX_ENABLED

// CHECK-DAG: @stdout = external
// CHECK-DAG: call {{.*}} @sysconf(i32 {{.*}}97)

void foo(uniform int out[]) {
    print("programIndex=%\n", programIndex);
    out[0] = num_cores();
}
