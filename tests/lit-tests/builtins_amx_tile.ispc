// This test checks that AMX tile (amxtile) builtins are correctly handled by ISPC

// RUN: not %{ispc} --nowrap %s --target=avx512icl-x4 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_ICL_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx512icl-x8 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_ICL_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx512icl-x16 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_ICL_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx512icl-x32 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_ICL_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx512icl-x64 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_ICL_ERROR

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

// RUN: not %{ispc} --nowrap %s --target=avx512icl-x4 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_ICL_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx512icl-x8 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_ICL_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx512icl-x16 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_ICL_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx512icl-x32 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_ICL_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx512icl-x64 --emit-llvm-text -O0 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_ICL_ERROR

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

// CHECK_ICL_ERROR: Error: Some AMX functions used in the source are not supported on the current target

// CHECK_SPR_SUCCESS: call void @llvm.x86.ldtilecfg
// CHECK_SPR_SUCCESS: call void @llvm.x86.sttilecfg
// CHECK_SPR_SUCCESS: call void @llvm.x86.tilezero
// CHECK_SPR_SUCCESS: call void @llvm.x86.tileloadd64
// CHECK_SPR_SUCCESS: call void @llvm.x86.tileloaddt164
// CHECK_SPR_SUCCESS: call void @llvm.x86.tilestored64
// CHECK_SPR_SUCCESS: call void @llvm.x86.tilerelease

// CHECK_GNR_SUCCESS: call void @llvm.x86.ldtilecfg
// CHECK_GNR_SUCCESS: call void @llvm.x86.sttilecfg
// CHECK_GNR_SUCCESS: call void @llvm.x86.tilezero
// CHECK_GNR_SUCCESS: call void @llvm.x86.tileloadd64
// CHECK_GNR_SUCCESS: call void @llvm.x86.tileloaddt164
// CHECK_GNR_SUCCESS: call void @llvm.x86.tilestored64
// CHECK_GNR_SUCCESS: call void @llvm.x86.tilerelease

// CHECK_DMR_SUCCESS: call void @llvm.x86.ldtilecfg
// CHECK_DMR_SUCCESS: call void @llvm.x86.sttilecfg
// CHECK_DMR_SUCCESS: call void @llvm.x86.tilezero
// CHECK_DMR_SUCCESS: call void @llvm.x86.tileloadd64
// CHECK_DMR_SUCCESS: call void @llvm.x86.tileloaddt164
// CHECK_DMR_SUCCESS: call void @llvm.x86.tilestored64
// CHECK_DMR_SUCCESS: call void @llvm.x86.tilerelease

#include <amx.isph>

constexpr uniform int tile_zero_id() { return amx_tmm2; }
constexpr uniform int tile_load_id() { return amx_tmm0; }
constexpr uniform int tile_load_t1_id() { return amx_tmm1; }

__attribute__((external_only))
export void test_amx_tile(uniform int8* uniform config, uniform int8* uniform data) {
    // Base AMX tile operations - require amxtile feature
    amx_tile_loadconfig(config);
    amx_tile_storeconfig(config);
    // Exercise constexpr + template AMX wrappers for tile immargs.
    amx_tile_zero<tile_zero_id()>();
    amx_tile_load<tile_load_id()>(data, (uniform int64)64);
    amx_tile_load_t1<tile_load_t1_id()>(data, (uniform int64)64);
    amx_tile_store<tile_load_id()>(data, (uniform int64)64);
    amx_tile_release();
}
