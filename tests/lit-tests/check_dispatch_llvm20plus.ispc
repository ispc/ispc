// Compile the test for different x86 targets available with LLVM_20+ with auto-dispatch enabled and run the test under SDE to verify that correct platform was picked in runtime.

// RUN: %{ispc} %s --target=sse2-i32x4,avx10.2dmr-x16 -o %t_ispc0.o --nostdlib
// RUN: %{cc} -O2 %S/check_dispatch.c %t_ispc0*.o -o %t.exe
// RUN: sde -mrm -- %t.exe | FileCheck %s -check-prefix=CHECK_SSE2
// RUN: sde -dmr -- %t.exe | FileCheck %s -check-prefix=CHECK_DMR

// REQUIRES: X86_ENABLED && SDE_INSTALLED && LLVM_20_0+

// CHECK_SSE2: SSE2
// CHECK_DMR: AVX10_2DMR

export uniform int detect_isa() {
#if defined(ISPC_TARGET_AVX10_2DMR)
    return 11;
#elif defined(ISPC_TARGET_AVX512GNR)
    return 10;
#elif defined(ISPC_TARGET_AVX512SPR)
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
