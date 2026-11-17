// Ordinary shift overloads should use packed instructions even with a runtime count.
// RUN: %{ispc} %s --target=avx2-i32x8 -I %S/../../stdlib/include -O2 --nowrap --emit-asm -o - | FileCheck %s
// REQUIRES: X86_ENABLED

#include "intrinsics/emmintrin.isph"

unmasked void left_words(uniform int32 src[], uniform int32 dst[], uniform int count) {
    uniform __m128i data = {{src[0], src[1], src[2], src[3]}};
    uniform __m128i shifted = _mm_slli_epi16(data, count);
    for (uniform int i = 0; i < 4; ++i) {
        dst[i] = shifted.m128i_i32[i];
    }
}

// CHECK-LABEL: {{^_?}}left_words___
// CHECK: vpsllw{{[[:blank:]]+}}%xmm

unmasked void right_signed_words(uniform int32 src[], uniform int32 dst[], uniform int count) {
    uniform __m128i data = {{src[0], src[1], src[2], src[3]}};
    uniform __m128i shifted = _mm_srai_epi16(data, count);
    for (uniform int i = 0; i < 4; ++i) {
        dst[i] = shifted.m128i_i32[i];
    }
}

// CHECK-LABEL: {{^_?}}right_signed_words___
// CHECK: vpsraw{{[[:blank:]]+}}%xmm

unmasked void right_quadwords(uniform int32 src[], uniform int32 dst[], uniform int count) {
    uniform __m128i data = {{src[0], src[1], src[2], src[3]}};
    uniform __m128i shifted = _mm_srli_epi64(data, count);
    for (uniform int i = 0; i < 4; ++i) {
        dst[i] = shifted.m128i_i32[i];
    }
}

// CHECK-LABEL: {{^_?}}right_quadwords___
// CHECK: vpsrlq{{[[:blank:]]+}}%xmm
