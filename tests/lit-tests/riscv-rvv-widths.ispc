// The gang of an rvv-<VLEN>b target is VLEN/32 lanes, the number of 32-bit
// elements in one vector register, so this loop over 32-bit elements runs
// with the whole gang in a single register at LMUL=1.

// RUN: %{ispc} --target=rvv-128b --arch=riscv64 --target-os=linux --emit-asm %s -o - | FileCheck %s --check-prefixes=CHECK,X4
// RUN: %{ispc} --target=rvv-256b --arch=riscv64 --target-os=linux --emit-asm %s -o - | FileCheck %s --check-prefixes=CHECK,X8
// RUN: %{ispc} --target=rvv-512b --arch=riscv64 --target-os=linux --emit-asm %s -o - | FileCheck %s --check-prefixes=CHECK,X16
// RUN: %{ispc} --target=rvv-1024b --arch=riscv64 --target-os=linux --emit-asm %s -o - | FileCheck %s --check-prefixes=CHECK,X32
// RUN: %{ispc} --target=rvv-2048b --arch=riscv64 --target-os=linux --emit-asm %s -o - | FileCheck %s --check-prefixes=CHECK,X64

// REQUIRES: RISCV_ENABLED && LINUX_ENABLED

// CHECK-LABEL: inc___
// X4: vsetivli zero, 4, e32, m1
// X8: vsetivli zero, 8, e32, m1
// X16: vsetivli zero, 16, e32, m1
// X32: vsetvli zero, {{.*}}, e32, m1
// X64: vsetvli zero, {{.*}}, e32, m1
// CHECK: vle32.v
// CHECK: vadd.vi
// CHECK: vse32.v

export void inc(uniform int v[], uniform int count) {
    foreach (i = 0 ... count) {
        v[i] += 1;
    }
}
