// Test basic foreach loop with array addition for RISC-V target

// RUN: %{ispc} --target=rvv-x4 --arch=riscv64 --cpu=spacemit-x60 --emit-asm %s -o - | FileCheck %s
// RUN: %{ispc} --target=rvv-x4 --arch=riscv64 --cpu=generic-rv64gcv --emit-asm %s -o - | FileCheck %s
// RUN: %{ispc} --target=rvv-x4 --arch=riscv64 --emit-asm %s -o - | FileCheck %s

// REQUIRES: RISCV_ENABLED && LLVM_19_0+

// CHECK-LABEL: simple_sum___un_3C_unf_3E_un_3C_unf_3E_un_3C_unf_3E_uni:
// CHECK: vsetivli
// CHECK: vle32.v
// CHECK: vle32.v
// CHECK: vfadd.vv
// CHECK: vse32.v

export void simple_sum(uniform float a[], uniform float b[], uniform float result[], uniform int count) {
    foreach (i = 0 ... count) {
        result[i] = a[i] + b[i];
    }
}
