// This test checks that AMX tile (amxtile) intrinsics are correctly handled by ISPC

// RUN: not %{ispc} --nowrap %s --target=avx512icl-x4 --enable-llvm-intrinsics --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_ICL_ERROR
// RUN: %{ispc} --nowrap %s --target=avx512spr-x4 --enable-llvm-intrinsics --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_SPR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx512spr-x4 --cpu=gnr --enable-llvm-intrinsics --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_GNR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx10.2dmr-x4 --cpu=dmr --enable-llvm-intrinsics --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_DMR_SUCCESS

// REQUIRES: X86_ENABLED && LLVM_20_0+ && !MACOS_HOST

// CHECK_ICL_ERROR: Target specific LLVM intrinsic "ldtilecfg" requires feature "amxtile" not supported on "icelake-client" CPU
// CHECK_ICL_ERROR: Target specific LLVM intrinsic "sttilecfg" requires feature "amxtile" not supported on "icelake-client" CPU
// CHECK_ICL_ERROR: Target specific LLVM intrinsic "tilezero" requires feature "amxtile" not supported on "icelake-client" CPU
// CHECK_ICL_ERROR: Target specific LLVM intrinsic "tileloadd64" requires feature "amxtile" not supported on "icelake-client" CPU
// CHECK_ICL_ERROR: Target specific LLVM intrinsic "tileloaddt164" requires feature "amxtile" not supported on "icelake-client" CPU
// CHECK_ICL_ERROR: Target specific LLVM intrinsic "tilestored64" requires feature "amxtile" not supported on "icelake-client" CPU
// CHECK_ICL_ERROR: Target specific LLVM intrinsic "tilerelease" requires feature "amxtile" not supported on "icelake-client" CPU

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

__attribute__((external_only))
export void test_amx_tile(uniform int8* uniform config, uniform int8* uniform data) {
    // Base AMX tile operations - require amxtile feature
    @llvm.x86.ldtilecfg(config);
    @llvm.x86.sttilecfg(config);
    @llvm.x86.tilezero((uniform int8)2);
    @llvm.x86.tileloadd64((uniform int8)0, data, (uniform int64)64);
    @llvm.x86.tileloaddt164((uniform int8)1, data, (uniform int64)64);
    @llvm.x86.tilestored64((uniform int8)0, data, (uniform int64)64);
    @llvm.x86.tilerelease();
}
