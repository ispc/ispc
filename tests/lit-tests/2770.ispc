// The test is checking that shuffle(T, T, int) is mapped to particular HW instruction on SSE4-AVX2 targets.

// RUN: %{ispc} --target=avx2-i32x8 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s
// RUN: %{ispc} --target=avx2vnni-i32x8 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s

// RUN: %{ispc} --target=avx2-i32x4 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s
// RUN: %{ispc} --target=avx2vnni-i32x4 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s
// RUN: %{ispc} --target=avx2-i64x4 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK-LABEL: Shuffle_i32___
// CHECK_AVX2x8: vpermd
// CHECK_AVX2x4: vpermilps
unmasked varying int32 Shuffle_i32(varying int32 a, varying int32 b, varying int32 perm) {
    return shuffle(a, b, perm);
}

// CHECK_ALL: Shuffle_float___
// CHECK_AVX2x8: vpermd
// CHECK_AVX2x4: vpermilps
unmasked varying float Shuffle_float(varying float a, varying float b, varying int32 perm) {
    return shuffle(a, b, perm);
}
