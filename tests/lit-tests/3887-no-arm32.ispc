// Disabling ARM32 must reject both NEON and generic ARM targets while retaining
// AArch64 with either addressing width.
// RUN: not %{ispc} %s --arch=arm --cpu=cortex-a35 --target=neon-i32x4 --target-os=linux --nowrap -o %t.o 2>&1 | FileCheck %s --check-prefix=ARM
// RUN: not %{ispc} %s --arch=arm --cpu=cortex-a35 --target=generic-i32x4 --target-os=linux --nowrap -o %t.o 2>&1 | FileCheck %s --check-prefix=ARM
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --target-os=linux --addressing=32 --nowrap -o %t.o
// RUN: %{ispc} %s --arch=aarch64 --target=generic-i32x4 --target-os=linux --addressing=64 --nowrap -o %t.o
// REQUIRES: ARM_ENABLED, LINUX_ENABLED, !ARM_32BIT_ENABLED

// ARM: target for arm on Linux is not supported in current build
void foo(uniform int out[]) {
    print("programIndex=%\n", programIndex);
    out[0] = num_cores();
}
