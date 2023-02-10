// This test does uint->float varying typecast and verifies that performance
// warning is printed only for relevant cases.

//; RUN: %{ispc} %s --emit-asm -o %t.sse2.s --nowrap --target=sse2-i32x4 2>&1 | FileCheck %s -check-prefix=CHECK_SSE2_WARNING
//; RUN: FileCheck --input-file=%t.sse2.s -check-prefix=CHECK_SSE2_ASM %s
//; RUN: %{ispc} %s --emit-asm -o %t.avx2.s --nowrap --target=avx2-i32x8 2>&1 | FileCheck %s -check-prefix=CHECK_AVX2_WARNING
//; RUN: FileCheck --input-file=%t.avx2.s -check-prefix=CHECK_AVX2_ASM %s
//; RUN: %{ispc} %s --emit-asm -o %t.avx512skx.s --nowrap --target=avx512skx-x16 2>&1 | FileCheck %s --allow-empty -check-prefix=CHECK_AVX512SKX_WARNING
//; RUN: FileCheck --input-file=%t.avx512skx.s -check-prefix=CHECK_AVX512SKX_ASM %s
//; RUN: %{ispc} %s --emit-asm -o %t.avx512knl.s --nowrap --target=avx512knl-x16 2>&1 | FileCheck %s --allow-empty -check-prefix=CHECK_AVX512KNL_WARNING
//; RUN: FileCheck --input-file=%t.avx512knl.s -check-prefix=CHECK_AVX512KNL_ASM %s

//; REQUIRES: X86_ENABLED

//; CHECK_SSE2_WARNING: Performance Warning: Conversion from uint32 to float is slow. Use "int32" if possible
//; CHECK_SSE2_ASM-NOT: vcvt

//; CHECK_AVX2_WARNING: Performance Warning: Conversion from uint32 to float is slow. Use "int32" if possible
//; CHECK_AVX2_ASM-NOT: vcvt

//; CHECK_AVX512SKX_WARNING-NOT: Performance Warning: Conversion from uint32 to float is slow. Use "int32" if possible
//; CHECK_AVX512SKX_ASM: vcvt

//; CHECK_AVX512KNL_WARNING-NOT: Performance Warning: Conversion from uint32 to float is slow. Use "int32" if possible
//; CHECK_AVX512KNL_ASM: vcvt


unmasked varying float test_uint_f(varying uint arg1)
{
    varying float ret = (varying float) arg1;
    return ret;

}
