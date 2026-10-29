// Verifies that the iOS C builtins use the Apple stdout symbol and
// _SC_NPROCESSORS_ONLN value. The builtins are built without the C library
// headers (#3887).
// See https://github.com/ispc/ispc/issues/3887

// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --target-os=ios --nowrap --emit-llvm-text -o - | FileCheck %s

// REQUIRES: ARM_ENABLED && MACOS_IOS_ENABLED

// CHECK-DAG: @__stdoutp = external
// CHECK-DAG: call {{.*}} @sysconf(i32 {{.*}}58)

void foo(uniform int out[]) {
    print("programIndex=%\n", programIndex);
    out[0] = num_cores();
}
