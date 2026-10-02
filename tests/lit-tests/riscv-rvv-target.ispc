// The rvv target is rvv-128b unless a zvl<N>b attribute asks for wider vector
// registers, in which case it is rvv-<N>b with a gang of N/32 lanes. rvv-x4 is
// the legacy name of rvv-128b. A device name never changes the gang by itself.

// RUN: %{ispc} --target=rvv --arch=riscv64 --target-os=linux --emit-asm %s -o - | FileCheck %s --check-prefixes=CHECK,X4
// RUN: %{ispc} --target=rvv --attr=zvl256b --arch=riscv64 --target-os=linux --emit-asm %s -o - | FileCheck %s --check-prefixes=CHECK,X8
// RUN: %{ispc} --target=rvv --attr=zvl512b --arch=riscv64 --target-os=linux --emit-asm %s -o - | FileCheck %s --check-prefixes=CHECK,X16
// RUN: %{ispc} --target=rvv --attr=zvl1024b --arch=riscv64 --target-os=linux --emit-asm %s -o - | FileCheck %s --check-prefixes=CHECK,X32
// RUN: %{ispc} --target=rvv-256b --arch=riscv64 --target-os=linux --emit-asm %s -o - | FileCheck %s --check-prefixes=CHECK,X8
// RUN: %{ispc} --target=rvv-256b --attr=zvl256b --arch=riscv64 --target-os=linux --emit-asm %s -o - | FileCheck %s --check-prefixes=CHECK,X8
// RUN: %{ispc} --target=rvv-x4 --arch=riscv64 --target-os=linux --emit-asm %s -o - | FileCheck %s --check-prefixes=CHECK,X4
// RUN: %{ispc} --target=rvv --cpu=spacemit-x60 --arch=riscv64 --target-os=linux --emit-asm %s -o - | FileCheck %s --check-prefixes=CHECK,X4-ON-X60
// RUN: %{ispc} --target=rvv --cpu=spacemit-x60 --attr=zvl256b --arch=riscv64 --target-os=linux --emit-asm %s -o - | FileCheck %s --check-prefixes=CHECK,X8
// RUN: %{ispc} --target=rvv --attr=zvl256b,zvfh --arch=riscv64 --target-os=linux --emit-asm %s -o - | FileCheck %s --check-prefixes=CHECK,X8

// REQUIRES: RISCV_ENABLED && LINUX_ENABLED

// CHECK-LABEL: inc___
// X4: vsetivli zero, 4, e32, m1
// X8: vsetivli zero, 8, e32, m1
// X16: vsetivli zero, 16, e32, m1
// X32: vsetvli zero, {{.*}}, e32, m1
// The device has 256-bit registers, so a 4-wide gang is half a register.
// X4-ON-X60: vsetivli zero, 4, e32, mf2
// CHECK: vle32.v
// CHECK: vadd.vi
// CHECK: vse32.v

export void inc(uniform int v[], uniform int count) {
    foreach (i = 0 ... count) {
        v[i] += 1;
    }
}
