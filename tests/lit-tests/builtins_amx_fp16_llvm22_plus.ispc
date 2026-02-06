// This test checks that AMX FP16 (amxfp16) builtins are correctly handled by ISPC on NVL target

// RUN: not %{ispc} --nowrap %s --target=avx10.2nvl-x4 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_NVL_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx10.2nvl-x8 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_NVL_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx10.2nvl-x16 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_NVL_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx10.2nvl-x32 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_NVL_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx10.2nvl-x64 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_NVL_ERROR

// REQUIRES: X86_ENABLED && LLVM_22_0+ && !MACOS_HOST

// CHECK_NVL_ERROR: Error: Some AMX functions used in the source are not supported on the current target

#include <amx.isph>

__attribute__((external_only))
export void test_amx_fp16() {
    // require amxfp16 feature (GNR+)
    amx_dpfp16ps(2, 0, 1);
}
