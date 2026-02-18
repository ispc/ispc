// RUN: %{ispc} %s --target=avx2-i32x8 --arch=x86-64 --addressing=64 --emit-asm -o - | FileCheck %s

// UNSUPPORTED: !LLVM_20_0+

// REQUIRES: X86_ENABLED

// The loop should be unrolled with 4 vmovups followed by 4 vfmadd instructions.
// LLVM register allocation may insert moves between fmadd instructions.
// CHECK:      vmovups (
// CHECK-NEXT: vmovups 32(
// CHECK-NEXT: vmovups 64(
// CHECK-NEXT: vmovups 96(
// CHECK-NEXT: vfmadd{{[0-9]+}}ps (
// CHECK-NEXT: vfmadd{{[0-9]+}}ps 32(
// CHECK-NEXT: vfmadd{{[0-9]+}}ps 64(
// CHECK: vfmadd{{[0-9]+}}ps 96(

unmasked uniform float dotProductSoA_Unroll(uniform float a[], uniform float b[], uniform size_t N)
{
    varying float dot = 0;
    
    #pragma unroll 4
    for(uniform size_t c = 0; c < N; c += programCount)
    {
        varying int i = c + programIndex;
        dot += a[i] * b[i];
    }

    return reduce_add(dot);
}
