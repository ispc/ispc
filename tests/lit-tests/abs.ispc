//; RUN: %{ispc} --emit-asm --target=avx2 %s -o - | FileCheck %s -check-prefix=CHECK_AVX2
//; RUN: %{ispc} --emit-asm --target=sse4 %s -o - | FileCheck %s -check-prefix=CHECK_SSE4
//; RUN: %{ispc} --emit-asm --target=avx512skx-i32x16 %s -o - | FileCheck %s -check-prefix=CHECK_AVX512
// REQUIRES: X86_ENABLED
unsigned int foo_var(int a) {

//; CHECK_SSE4: pabsd
//; CHECK_AVX2: vpabsd
//; CHECK_AVX512: vpabsd
    return abs(a);

}
