// Checks the target libc sysconf constant and header-independent print() (#3887).

// RUN: %{ispc} %s --arch=x86-64 --target=sse2-i32x4 --target-os=linux --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --arch=x86 --target=sse2-i32x4 --target-os=linux --nowrap --emit-llvm-text -o - | FileCheck %s

// REQUIRES: X86_ENABLED && LINUX_ENABLED && !ANDROID_HOST

// CHECK-DAG: @stdout = external
// CHECK-DAG: call {{.*}} @fputs(
// CHECK-DAG: call {{.*}} @fflush(
// CHECK-DAG: call {{.*}} @sysconf(i32 {{[^0-9-]*}}84)

export void foo(uniform int out[]) {
    print("programIndex=%\n", programIndex);
    out[0] = num_cores();
}
