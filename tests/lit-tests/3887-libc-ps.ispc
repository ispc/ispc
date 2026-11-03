// Checks the FreeBSD-derived PlayStation libc constant and stdout symbol;
// the unrecognized x86_64-scei-ps triple selected the glibc ABI instead.
// RUN: %{ispc} %s --arch=x86-64 --target=sse2-i32x4 --target-os=ps4 --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --arch=x86-64 --target=sse2-i32x4 --target-os=ps5 --nowrap --emit-llvm-text -o - | FileCheck %s
// REQUIRES: X86_ENABLED && PS_ENABLED

// CHECK-DAG: @__stdoutp = external
// CHECK-DAG: call {{.*}} @sysconf(i32 {{[^0-9-]*}}58)

uniform int foo() {
    print("programIndex=%\n", programIndex);
    return num_cores();
}
