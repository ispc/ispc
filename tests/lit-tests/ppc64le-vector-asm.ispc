// Test that PowerPC VSX vector instructions are generated for ppc64le targets.

// RUN: %{ispc} %s --target=generic-i32x4 --arch=ppc64le --target-os=linux --nowrap --emit-asm -o - | FileCheck %s

// REQUIRES: PPC64_ENABLED && LINUX_ENABLED

// CHECK-LABEL: vadd___un_3C_unf_3E_un_3C_unf_3E_un_3C_unf_3E_uni:
// CHECK: xvaddsp

void vadd(uniform float a[], uniform float b[], uniform float result[], uniform int count) {
    foreach (i = 0 ... count) {
        result[i] = a[i] + b[i];
    }
}

// CHECK-LABEL: vmul___un_3C_unf_3E_un_3C_unf_3E_un_3C_unf_3E_uni:
// CHECK: xvmulsp

void vmul(uniform float a[], uniform float b[], uniform float result[], uniform int count) {
    foreach (i = 0 ... count) {
        result[i] = a[i] * b[i];
    }
}
