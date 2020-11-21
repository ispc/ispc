//; RUN: %{ispc} %s --target=avx2-i32x8 --emit-asm -o - | FileCheck %s --implicit-check-not "vmovups" -check-prefix=CHECK-ASM
//; RUN: %{ispc} %s --target=avx2-i32x8 --emit-llvm-text -O0 -o - | FileCheck %s --implicit-check-not "align 1"
//; REQUIRES: X86_ENABLED

typedef float<TARGET_WIDTH> AlignedFloat;

//; CHECK-ASM: vmovaps
void Aligned_Load(uniform float Result[], const uniform float Source1[], const uniform float Source2[], const uniform unsigned int Iterations)
{
    // Loading a uniform pointer to a non atomic type results in aligned load.
    // This is accomplished by not setting alignment explicitly = 1 in IR for LoadInst.
    const uniform AlignedFloat S1 = *((uniform AlignedFloat *uniform)&Source1[3]);
    const uniform AlignedFloat S2 = *((uniform AlignedFloat *uniform)&Source2[3]);

	const uniform AlignedFloat R = S1 + S2;
    *((uniform AlignedFloat *uniform)&Result[3]) = *((uniform AlignedFloat *uniform)&R);
}
