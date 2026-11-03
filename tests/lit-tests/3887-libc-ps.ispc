// Checks the FreeBSD-derived PlayStation libc constant; the unrecognized
// x86_64-scei-ps triple selected the glibc value instead.
// RUN: %{ispc} %s --arch=x86-64 --target=sse2-i32x4 --target-os=ps4 --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --arch=x86-64 --target=sse2-i32x4 --target-os=ps5 --nowrap --emit-llvm-text -o - | FileCheck %s
// REQUIRES: X86_ENABLED && PS_ENABLED

// CHECK: call {{.*}} @sysconf(i32 {{.*}}58)

uniform int foo() { return num_cores(); }
