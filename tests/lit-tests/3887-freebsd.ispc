// Verifies that the FreeBSD C builtins use the FreeBSD stdout symbol and
// _SC_NPROCESSORS_ONLN value, also when ISPC is built with another C library's
// headers (e.g. glibc on Linux).
// See https://github.com/ispc/ispc/issues/3887

// RUN: %{ispc} %s --arch=x86-64 --target=avx2-i32x8 --target-os=freebsd --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --arch=x86 --target=avx2-i32x8 --target-os=freebsd --nowrap --emit-llvm-text -o - | FileCheck %s

// REQUIRES: X86_ENABLED && FREEBSD_ENABLED

// CHECK-DAG: @__stdoutp = external
// CHECK-DAG: call {{.*}} @sysconf(i32 {{.*}}58)

void foo(uniform int out[]) {
    print("programIndex=%\n", programIndex);
    out[0] = num_cores();
}
