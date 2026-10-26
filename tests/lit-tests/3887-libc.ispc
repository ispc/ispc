// Verifies that the C builtins use the stdout symbol and the _SC_NPROCESSORS_ONLN
// value of the C library of the target OS: glibc on Linux, bionic on Android,
// and the FreeBSD C library on FreeBSD. The builtins are built without the C
// library headers, so the values do not depend on the build host.
// See https://github.com/ispc/ispc/issues/3887

// RUN: %{ispc} %s --arch=x86-64 --target=avx2-i32x8 --target-os=linux --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=LINUX
// RUN: %{ispc} %s --arch=x86-64 --target=avx2-i32x8 --target-os=android --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=ANDROID
// RUN: %{ispc} %s --arch=x86-64 --target=avx2-i32x8 --target-os=freebsd --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=FREEBSD
// RUN: %{ispc} %s --arch=x86 --target=avx2-i32x8 --target-os=freebsd --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=FREEBSD

// REQUIRES: X86_ENABLED && LINUX_ENABLED && ANDROID_ENABLED && FREEBSD_ENABLED

// LINUX-DAG: @stdout = external
// LINUX-DAG: call {{.*}} @sysconf(i32 {{.*}}84)

// ANDROID-DAG: @stdout = external
// ANDROID-DAG: call {{.*}} @sysconf(i32 {{.*}}97)

// FREEBSD-DAG: @__stdoutp = external
// FREEBSD-DAG: call {{.*}} @sysconf(i32 {{.*}}58)

void foo(uniform int out[]) {
    print("programIndex=%\n", programIndex);
    out[0] = num_cores();
}
