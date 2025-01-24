// Compile the test for different x86 targets with auto-dispatch enabled and run the test under SDE to verify that correct platform was picked in runtime.

// RUN: %{ispc} %s --target=sse2-i32x4,sse4-i32x4,avx1-i32x8,avx2-i32x8,avx2vnni-i32x8,avx512skx-x16,avx512icl-x16,avx512spr-x16 -o %t_ispc0.o --nostdlib
// RUN: %{cc} -O2 %S/check_dispatch.c %t_ispc0*.o -o %t.exe
// RUN: sde -mrm -- %t.exe | FileCheck %s -check-prefix=CHECK_SSE2
// Note sse4-i32x4 implies SSE4.2 ISA, while Penryn is SSE4.1, so should fall back to SSE2.
// RUN: sde -pnr -- %t.exe | FileCheck %s -check-prefix=CHECK_SSE2
// RUN: sde -nhm -- %t.exe | FileCheck %s -check-prefix=CHECK_SSE4
// RUN: sde -snb -- %t.exe | FileCheck %s -check-prefix=CHECK_AVX1
// RUN: sde -hsw -- %t.exe | FileCheck %s -check-prefix=CHECK_AVX2
// RUN: sde -adl -- %t.exe | FileCheck %s -check-prefix=CHECK_AVX2VNNI
// RUN: sde -skx -- %t.exe | FileCheck %s -check-prefix=CHECK_SKX
// RUN: sde -icl -- %t.exe | FileCheck %s -check-prefix=CHECK_ICL
// RUN: sde -spr -- %t.exe | FileCheck %s -check-prefix=CHECK_SPR

// RUN: %{ispc} %s --target=sse2-i32x4,sse4.1-i32x4,avx1-i32x8,avx2-i32x8,avx512skx-x16,avx512spr-x16 -o %t_ispc2.o --nostdlib
// RUN: %{cc} -O2 %S/check_dispatch.c %t_ispc2*.o -o %t2.exe
// RUN: sde -mrm -- %t2.exe | FileCheck %s -check-prefix=CHECK_SSE2
// sse4.1-i32x4 is specified, so both Penryn and Nehalem should use it.
// RUN: sde -pnr -- %t2.exe | FileCheck %s -check-prefix=CHECK_SSE4
// RUN: sde -nhm -- %t2.exe | FileCheck %s -check-prefix=CHECK_SSE4

// RUN: %{ispc} %s --target=sse2-i32x4,sse4.2-i32x4,avx1-i32x8,avx2-i32x8,avx512skx-x16,avx512spr-x16 -o %t_ispc3.o --nostdlib
// RUN: %{cc} -O2 %S/check_dispatch.c %t_ispc3*.o -o %t3.exe
// RUN: sde -mrm -- %t3.exe | FileCheck %s -check-prefix=CHECK_SSE2
// sse4.2-i32x4 is specified, so Penryn should fall back to SSE2.
// RUN: sde -pnr -- %t3.exe | FileCheck %s -check-prefix=CHECK_SSE2
// RUN: sde -nhm -- %t3.exe | FileCheck %s -check-prefix=CHECK_SSE4

// REQUIRES: X86_ENABLED && SDE_INSTALLED

// CHECK_SSE2: SSE2
// CHECK_SSE4: SSE4
// CHECK_AVX1: AVX1
// CHECK_AVX2: AVX2
// CHECK_AVX2VNNI: AVX2VNNI
// CHECK_SKX: AVX512SKX
// CHECK_ICL: AVX512ICL
// CHECK_SPR: AVX512SPR

export uniform int detect_isa() {
#if defined(ISPC_TARGET_AVX512SPR)
    return 9;
#elif defined(ISPC_TARGET_AVX512ICL)
    return 8;
#elif defined(ISPC_TARGET_AVX512SKX)
    return 7;
#elif defined(ISPC_TARGET_AVX2VNNI)
    return 5;
#elif defined(ISPC_TARGET_AVX2)
    return 4;
#elif defined(ISPC_TARGET_AVX)
    return 3;
#elif defined(ISPC_TARGET_SSE4)
    return 2;
#elif defined(ISPC_TARGET_SSE2)
    return 1;
#else
    return 0;
#endif
}
