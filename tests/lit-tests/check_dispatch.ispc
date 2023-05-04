// Compile the test for different x86 targets with auto-dispatch enabled and run the test under SDE to verify that correct platform was picked in runtime.

// RUN: %{ispc} %s --target=sse2-i32x4,sse4-i32x4,avx1-i32x8,avx2-i32x8,avx512knl-x16,avx512skx-x16,avx512spr-x16 -o %t_ispc.o --nostdlib
// RUN: %{cc} -O2 %S/check_dispatch.c %t_ispc*.o -o check_dispatch.exe
// RUN: sde -mrm -- check_dispatch.exe | FileCheck %s -check-prefix=CHECK_SSE2
// Note sse4-i32x4 implies SSE4.2 ISA, while Penryn is SSE4.1, so should fall back to SSE2.
// RUN: sde -pnr -- check_dispatch.exe | FileCheck %s -check-prefix=CHECK_SSE2
// RUN: sde -nhm -- check_dispatch.exe | FileCheck %s -check-prefix=CHECK_SSE4
// RUN: sde -snb -- check_dispatch.exe | FileCheck %s -check-prefix=CHECK_AVX1
// RUN: sde -hsw -- check_dispatch.exe | FileCheck %s -check-prefix=CHECK_AVX2
// RUN: sde -skx -- check_dispatch.exe | FileCheck %s -check-prefix=CHECK_SKX
// RUN: sde -knl -- check_dispatch.exe | FileCheck %s -check-prefix=CHECK_KNL
// RUN: sde -spr -- check_dispatch.exe | FileCheck %s -check-prefix=CHECK_SPR

// RUN: %{ispc} %s --target=sse2-i32x4,sse4.1-i32x4,avx1-i32x8,avx2-i32x8,avx512knl-x16,avx512skx-x16,avx512spr-x16 -o %t_ispc.o --nostdlib
// RUN: %{cc} -O2 %S/check_dispatch.c %t_ispc*.o -o check_dispatch.exe
// RUN: sde -mrm -- check_dispatch.exe | FileCheck %s -check-prefix=CHECK_SSE2
// sse4.1-i32x4 is specified, so both Penryn and Nehalem should use it.
// RUN: sde -pnr -- check_dispatch.exe | FileCheck %s -check-prefix=CHECK_SSE4
// RUN: sde -nhm -- check_dispatch.exe | FileCheck %s -check-prefix=CHECK_SSE4

// RUN: %{ispc} %s --target=sse2-i32x4,sse4.2-i32x4,avx1-i32x8,avx2-i32x8,avx512knl-x16,avx512skx-x16,avx512spr-x16 -o %t_ispc.o --nostdlib
// RUN: %{cc} -O2 %S/check_dispatch.c %t_ispc*.o -o check_dispatch.exe
// RUN: sde -mrm -- check_dispatch.exe | FileCheck %s -check-prefix=CHECK_SSE2
// sse4.2-i32x4 is specified, so Penryn should fall back to SSE2.
// RUN: sde -pnr -- check_dispatch.exe | FileCheck %s -check-prefix=CHECK_SSE2
// RUN: sde -nhm -- check_dispatch.exe | FileCheck %s -check-prefix=CHECK_SSE4

// REQUIRES: X86_ENABLED && SDE_INSTALLED

// CHECK_SSE2: SSE2
// CHECK_SSE4: SSE4
// CHECK_AVX1: AVX1
// CHECK_AVX2: AVX2
// CHECK_SKX: SKX
// CHECK_KNL: KNL
// CHECK_SPR: SPR

export uniform int detect_isa() {
#if defined(ISPC_TARGET_AVX512SPR)
    return 7;
#elif defined(ISPC_TARGET_AVX512SKX)
    return 6;
#elif defined(ISPC_TARGET_AVX512KNL)
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
