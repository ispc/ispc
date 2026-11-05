// Checks the target libc sysconf constant and header-independent print() (#3887).

// RUN: %{ispc} %s --arch=x86-64 --target=sse2-i32x4 --target-os=android --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --arch=x86 --target=sse2-i32x4 --target-os=android --nowrap --emit-llvm-text -o - | FileCheck %s

// REQUIRES: X86_ENABLED && ANDROID_ENABLED

// CHECK-DAG: @stdout = external
// CHECK-DAG: call {{.*}} @fputs(
// CHECK-DAG: call {{.*}} @fflush(
// CHECK-DAG: call {{.*}} @sysconf(i32 {{[^0-9-]*}}97)

void foo(uniform int out[]) {
    print("programIndex=%\n", programIndex);
    out[0] = num_cores();
}
