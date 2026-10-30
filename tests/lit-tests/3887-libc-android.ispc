// Verifies that the Android C builtins use the bionic stdout symbol and
// _SC_NPROCESSORS_ONLN value on any build host. The builtins are built without
// the C library headers (#3887).
// See https://github.com/ispc/ispc/issues/3887

// RUN: %{ispc} %s --arch=x86-64 --target=sse2-i32x4 --target-os=android --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --arch=x86 --target=sse2-i32x4 --target-os=android --nowrap --emit-llvm-text -o - | FileCheck %s

// REQUIRES: X86_ENABLED && ANDROID_ENABLED

// CHECK-DAG: @stdout = external
// CHECK-DAG: call {{.*}} @sysconf(i32 {{.*}}97)

void foo(uniform int out[]) {
    print("programIndex=%\n", programIndex);
    out[0] = num_cores();
}
