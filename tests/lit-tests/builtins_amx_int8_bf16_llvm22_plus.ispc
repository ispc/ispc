// This test checks that AMX INT8 (amxint8) and BF16 builtins are correctly handled by ISPC on NVL target

// RUN: not %{ispc} --nowrap %s --target=avx10.2nvl-x4 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_NVL_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx10.2nvl-x8 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_NVL_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx10.2nvl-x16 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_NVL_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx10.2nvl-x32 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_NVL_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx10.2nvl-x64 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_NVL_ERROR

// REQUIRES: X86_ENABLED && LLVM_22_0+ && !MACOS_HOST

// CHECK_NVL_ERROR: Error: Some AMX functions used in the source are not supported on the current target

#include <amx.isph>

__attribute__((external_only))
export void test_amx_int8() {
    // require amxint8 feature (SPR+)
    amx_dpbssd(2, 0, 1);
    amx_dpbsud(2, 0, 1);
    amx_dpbusd(2, 0, 1);
    amx_dpbuud(2, 0, 1);
}

__attribute__((external_only))
export void test_amx_bf16() {
    // require amxbf16 feature (SPR+)
    amx_dpbf16ps(2, 0, 1);
}
