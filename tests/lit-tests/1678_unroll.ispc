// RUN: %{ispc} %s --target=avx2-i32x8 --arch=x86-64 --nostdlib --emit-asm -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// The goal of this test is to check that code generation for both versions is the same.

struct FVector4 {
    float<4> V;
};

struct WideFVector4 {
    float V[programCount];
};

unmasked inline uniform WideFVector4 operator+(const uniform WideFVector4 &A, const uniform WideFVector4 &B) {
    uniform WideFVector4 Result;
    Result.V[programIndex] = A.V[programIndex] + B.V[programIndex];
    return Result;
}

unmasked inline void LoadWideFVector4(uniform FVector4 *uniform DstPtr, const uniform FVector4 *uniform SrcPtr) {
    *DstPtr = *SrcPtr;
    *(DstPtr + 1) = *(SrcPtr + 1);
}

unmasked inline void StoreWideFVector4(uniform FVector4 *uniform DstPtr, const uniform FVector4 *uniform SrcPtr) {
    *DstPtr = *SrcPtr;
    *(DstPtr + 1) = *(SrcPtr + 1);
}

// CHECK-LABEL: AddWide___
// CHECK-COUNT-1: vmovups
// CHECK-NOT: vmovups
// CHECK-COUNT-1: vaddps
// CHECK-COUNT-1: vmovups
// CHECK-NOT: vmovups
// CHECK-COUNT-1: vzeroupper
unmasked void AddWide(uniform FVector4 Result[], uniform FVector4 Source1[], uniform FVector4 Source2[]) {
    uniform int Index = 0;
    uniform WideFVector4 S1, S2;

    LoadWideFVector4((uniform FVector4 * uniform) & S1, (uniform FVector4 * uniform) & Source1[Index]);
    LoadWideFVector4((uniform FVector4 * uniform) & S2, (uniform FVector4 * uniform) & Source2[Index]);
    const uniform WideFVector4 R = S1 + S2;
    StoreWideFVector4((uniform FVector4 * uniform) & Result[Index], (uniform FVector4 * uniform) & R);
}

unmasked inline void LoadWideFVector4_2(uniform FVector4 *uniform DstPtr, const uniform FVector4 *uniform SrcPtr) {
    for (uniform int i = 0; i < (programCount / 4); i++) {
        *(DstPtr + i) = *(SrcPtr + i);
    }
}

unmasked inline void StoreWideFVector4_2(uniform FVector4 *uniform DstPtr, const uniform FVector4 *uniform SrcPtr) {
    for (uniform int i = 0; i < (programCount / 4); i++) {
        *(DstPtr + i) = *(SrcPtr + i);
    }
}

// CHECK-LABEL: AddWide_2___
// CHECK-COUNT-1: vmovups
// CHECK-NOT: vmovups
// CHECK-COUNT-1: vaddps
// CHECK-COUNT-1: vmovups
// CHECK-NOT: vmovups
// CHECK-COUNT-1: vzeroupper
unmasked void AddWide_2(uniform FVector4 Result[], uniform FVector4 Source1[], uniform FVector4 Source2[]) {
    uniform int Index = 0;
    uniform WideFVector4 S1, S2;

    LoadWideFVector4_2((uniform FVector4 * uniform) & S1, (uniform FVector4 * uniform) & Source1[Index]);
    LoadWideFVector4_2((uniform FVector4 * uniform) & S2, (uniform FVector4 * uniform) & Source2[Index]);
    const uniform WideFVector4 R = S1 + S2;
    StoreWideFVector4_2((uniform FVector4 * uniform) & Result[Index], (uniform FVector4 * uniform) & R);
}
