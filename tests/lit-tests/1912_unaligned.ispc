//; RUN: %{ispc} %s --target=avx2-i32x8 --emit-asm -o - | FileCheck %s -check-prefix=CHECK-ASM
//; RUN: %{ispc} %s --target=avx2-i32x8 --emit-llvm-text -O0 -o - | FileCheck %s -check-prefix=CHECK-IR
//; REQUIRES: X86_ENABLED

typedef float<TARGET_WIDTH> AlignedFloat;

//; CHECK-ASM: vmovups

//; CHECK-IR: align 1

void UnAligned_Load(uniform float Result[], const uniform float Source1[], const uniform float Source2[])
{
    // Loading a uniform pointer to a varying atomic type results in unaligned load.
    // This is accomplished by setting alignment = 1 in IR for LoadInst.
    const varying float S1 = *((varying float *uniform)&Source1[3]);
    const varying float S2 = *((varying float *uniform)&Source2[3]);
    const varying float R = S1 + S2;

    *((varying float *uniform)&Result[3]) = *((varying float *uniform)&R);
}
