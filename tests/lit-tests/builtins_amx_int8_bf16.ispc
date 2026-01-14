// This test checks that AMX INT8 (amxint8) and BF16 builtins are correctly handled by ISPC

// RUN: %{ispc} --nowrap %s --target=avx512icl-x4 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_ICL_ERROR
// RUN: %{ispc} --nowrap %s --target=avx512icl-x8 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_ICL_ERROR
// RUN: %{ispc} --nowrap %s --target=avx512icl-x16 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_ICL_ERROR
// RUN: %{ispc} --nowrap %s --target=avx512icl-x32 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_ICL_ERROR
// RUN: %{ispc} --nowrap %s --target=avx512icl-x64 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_ICL_ERROR

// RUN: %{ispc} --nowrap %s --target=avx512spr-x4 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_SPR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx512spr-x8 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_SPR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx512spr-x16 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_SPR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx512spr-x32 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_SPR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx512spr-x64 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_SPR_SUCCESS

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

// RUN: %{ispc} --nowrap %s --target=avx512icl-x4 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_ICL_ERROR
// RUN: %{ispc} --nowrap %s --target=avx512icl-x8 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_ICL_ERROR
// RUN: %{ispc} --nowrap %s --target=avx512icl-x16 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_ICL_ERROR
// RUN: %{ispc} --nowrap %s --target=avx512icl-x32 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_ICL_ERROR
// RUN: %{ispc} --nowrap %s --target=avx512icl-x64 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_ICL_ERROR

// RUN: %{ispc} --nowrap %s --target=avx512spr-x4 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_SPR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx512spr-x8 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_SPR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx512spr-x16 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_SPR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx512spr-x32 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_SPR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx512spr-x64 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_SPR_SUCCESS

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

// CHECK_ICL_ERROR: @__not_supported()
// CHECK_ICL_ERROR: @__not_supported()
// CHECK_ICL_ERROR: @__not_supported()
// CHECK_ICL_ERROR: @__not_supported()
// CHECK_ICL_ERROR: @__not_supported()

// CHECK_SPR_SUCCESS: call void @llvm.x86.tdpbssd
// CHECK_SPR_SUCCESS: call void @llvm.x86.tdpbsud
// CHECK_SPR_SUCCESS: call void @llvm.x86.tdpbusd
// CHECK_SPR_SUCCESS: call void @llvm.x86.tdpbuud
// CHECK_SPR_SUCCESS: call void @llvm.x86.tdpbf16ps

// CHECK_GNR_SUCCESS: call void @llvm.x86.tdpbssd
// CHECK_GNR_SUCCESS: call void @llvm.x86.tdpbsud
// CHECK_GNR_SUCCESS: call void @llvm.x86.tdpbusd
// CHECK_GNR_SUCCESS: call void @llvm.x86.tdpbuud
// CHECK_GNR_SUCCESS: call void @llvm.x86.tdpbf16ps

// CHECK_DMR_SUCCESS: call void @llvm.x86.tdpbssd
// CHECK_DMR_SUCCESS: call void @llvm.x86.tdpbsud
// CHECK_DMR_SUCCESS: call void @llvm.x86.tdpbusd
// CHECK_DMR_SUCCESS: call void @llvm.x86.tdpbuud
// CHECK_DMR_SUCCESS: call void @llvm.x86.tdpbf16ps

#include <amx.isph>

__attribute__((external_only))
export void test_amx_int8() {
    // require amxint8 feature (SPR+)
    amx_dpbssd((uniform int8)2, (uniform int8)0, (uniform int8)1);
    amx_dpbsud((uniform int8)2, (uniform int8)0, (uniform int8)1);
    amx_dpbusd((uniform int8)2, (uniform int8)0, (uniform int8)1);
    amx_dpbuud((uniform int8)2, (uniform int8)0, (uniform int8)1);
}

__attribute__((external_only))
export void test_amx_bf16() {
    // require amxbf16 feature (SPR+)
    amx_dpbf16ps((uniform int8)2, (uniform int8)0, (uniform int8)1);
}
