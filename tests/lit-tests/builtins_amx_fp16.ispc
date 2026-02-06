// This test checks that AMX FP16 (amxfp16) builtins are correctly handled by ISPC

// RUN: not %{ispc} --nowrap %s --target=avx512spr-x4 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_SPR_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx512spr-x8 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_SPR_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx512spr-x16 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_SPR_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx512spr-x32 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_SPR_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx512spr-x64 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_SPR_ERROR

// RUN: %{ispc} --nowrap %s --target=avx512gnr-x4 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_GNR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx512gnr-x8 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_GNR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx512gnr-x16 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_GNR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx512gnr-x32 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_GNR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx512gnr-x64 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_GNR_SUCCESS

// RUN: %{ispc} --nowrap %s --target=avx10.2dmr-x4 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_DMR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx10.2dmr-x8 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_DMR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx10.2dmr-x16 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_DMR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx10.2dmr-x32 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_DMR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx10.2dmr-x64 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_DMR_SUCCESS

// RUN: not %{ispc} --nowrap %s --target=avx512spr-x4 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_SPR_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx512spr-x8 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_SPR_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx512spr-x16 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_SPR_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx512spr-x32 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_SPR_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx512spr-x64 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_SPR_ERROR

// RUN: %{ispc} --nowrap %s --target=avx512gnr-x4 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_GNR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx512gnr-x8 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_GNR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx512gnr-x16 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_GNR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx512gnr-x32 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_GNR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx512gnr-x64 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_GNR_SUCCESS

// RUN: %{ispc} --nowrap %s --target=avx10.2dmr-x4 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_DMR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx10.2dmr-x8 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_DMR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx10.2dmr-x16 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_DMR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx10.2dmr-x32 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_DMR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx10.2dmr-x64 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_DMR_SUCCESS

// REQUIRES: X86_ENABLED && LLVM_20_0+ && !MACOS_HOST

// CHECK_SPR_ERROR: Error: Some AMX functions used in the source are not supported on the current target
// CHECK_GNR_SUCCESS: call void @llvm.x86.tdpfp16ps
// CHECK_DMR_SUCCESS: call void @llvm.x86.tdpfp16ps

#include <amx.isph>

__attribute__((external_only))
export void test_amx_fp16() {
    // require amxfp16 feature (GNR+)
    amx_dpfp16ps(2, 0, 1);
}

