// RUN: %{ispc} %s --target=avx2-i64x4 --emit-asm -o - | FileCheck %s -check-prefix=CHECKAVX2_I64X4
// RUN: %{ispc} %s --target=avx2-i32x8 --emit-asm -o - | FileCheck %s -check-prefix=CHECKAVX2_I32X8
// RUN: %{ispc} %s --target=sse2-i32x4 --emit-asm -o - | FileCheck %s -check-prefix=CHECKSSE2_I32X4
// RUN: %{ispc} %s --target=sse2-i32x8 --emit-asm -o - | FileCheck %s -check-prefix=CHECKSSE2_I32X8
// RUN: %{ispc} %s --target=sse4-i16x8 --emit-asm -o - | FileCheck %s -check-prefix=CHECKSSE4_I16X8
// RUN: %{ispc} %s --target=sse4-i8x16 --emit-asm -o - | FileCheck %s -check-prefix=CHECKSSE4_I8X16
// RUN: %{ispc} %s --target=sse4.1-i16x8 --emit-asm -o - | FileCheck %s -check-prefix=CHECKSSE4_I16X8
// RUN: %{ispc} %s --target=sse4.1-i8x16 --emit-asm -o - | FileCheck %s -check-prefix=CHECKSSE4_I8X16

// REQUIRES: X86_ENABLED

// CHECKAVX2_I32X8: {{[a-z]*xor[a-z]*}}  %xmm0, %xmm0, %xmm0
// CHECKAVX2_I64X4: {{[a-z]*xor[a-z]*}}  %xmm0, %xmm0, %xmm0
// CHECKSSE2_I32X4: {{[a-z]*xor[a-z]*}}  %xmm0, %xmm0
// CHECKSSE2_I32X8: {{[a-z]*xor[a-z]*}}   %xmm0, %xmm0
// CHECKSSE4_I16X8: {{[a-z]*xor[a-z]*}}   %xmm0, %xmm0
// CHECKSSE4_I8X16: {{[a-z]*xor[a-z]*}}   %xmm0, %xmm0

extern uniform unsigned int var_18;
extern uniform unsigned int16 arr_202 [17] ;
extern uniform int8 arr_11 ;

void test() {
    foreach(i_33 = 0...16)
        arr_202 [i_33] = arr_11 ? (var_18 << (varying unsigned int64) 17) :  1U;
}
