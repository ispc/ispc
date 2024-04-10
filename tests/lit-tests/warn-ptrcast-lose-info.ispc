// RUN: %{ispc} --target=avx2-i32x8 --arch=x86-64 --nowrap --nostdlib %s --emit-llvm-text -o - 2>&1 | FileCheck %s
// RUN: %{ispc} --target=neon-i32x4 --arch=aarch64 --nowrap --nostdlib %s --emit-llvm-text -o - 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED
// REQUIRES: ARM_ENABLED

// CHECK: Warning: Pointer type cast of type "uniform int32 * uniform" to integer type "uniform int32" may lose information.

int32 foo(int * uniform x)  {
    return (int32) x;
}

