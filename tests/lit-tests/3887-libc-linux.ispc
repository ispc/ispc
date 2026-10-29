// Verifies that the Linux C builtins use the glibc stdout symbol and
// _SC_NPROCESSORS_ONLN value on any build host. The builtins are built without
// the C library headers (#3887). On Android hosts (Termux) the Linux target uses
// bionic, see 3887-libc-termux.ispc.
// See https://github.com/ispc/ispc/issues/3887

// RUN: %{ispc} %s --arch=x86-64 --target=sse2-i32x4 --target-os=linux --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --arch=x86 --target=sse2-i32x4 --target-os=linux --nowrap --emit-llvm-text -o - | FileCheck %s

// REQUIRES: X86_ENABLED && LINUX_ENABLED && !ANDROID_HOST

// CHECK-DAG: @stdout = external
// CHECK-DAG: call {{.*}} @sysconf(i32 {{.*}}84)

void foo(uniform int out[]) {
    print("programIndex=%\n", programIndex);
    out[0] = num_cores();
}
