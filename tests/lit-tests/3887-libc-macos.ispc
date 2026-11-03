// Checks the target libc sysconf constant and header-independent print() (#3887).

// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --target-os=macos --nowrap --emit-llvm-text -o - | FileCheck %s

// REQUIRES: ARM_ENABLED && MACOS_ARM_ENABLED

// CHECK-DAG: @__stdoutp = external
// CHECK-DAG: call {{.*}} @fputs(
// CHECK-DAG: call {{.*}} @fflush(
// CHECK-DAG: call {{.*}} @sysconf(i32 {{.*}}58)

void foo(uniform int out[]) {
    print("programIndex=%\n", programIndex);
    out[0] = num_cores();
}
