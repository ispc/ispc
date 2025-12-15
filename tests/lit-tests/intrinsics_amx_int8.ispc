// This test checks that AMX INT8 (amxint8) intrinsics are correctly handled by ISPC

// RUN: not %{ispc} --nowrap %s --target=avx512icl-x4 --enable-llvm-intrinsics --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_ICL_ERROR
// RUN: %{ispc} --nowrap %s --target=avx512spr-x4 --enable-llvm-intrinsics --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_SPR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx512gnr-x4 --enable-llvm-intrinsics --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_GNR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx10.2dmr-x4 --enable-llvm-intrinsics --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_DMR_SUCCESS

// REQUIRES: X86_ENABLED && LLVM_20_0+ && !MACOS_HOST

// CHECK_ICL_ERROR: Target specific LLVM intrinsic "tdpbssd" requires feature "amxint8" not supported on "icelake-client" CPU
// CHECK_ICL_ERROR: Target specific LLVM intrinsic "tdpbsud" requires feature "amxint8" not supported on "icelake-client" CPU
// CHECK_ICL_ERROR: Target specific LLVM intrinsic "tdpbusd" requires feature "amxint8" not supported on "icelake-client" CPU
// CHECK_ICL_ERROR: Target specific LLVM intrinsic "tdpbuud" requires feature "amxint8" not supported on "icelake-client" CPU

// CHECK_SPR_SUCCESS: call void @llvm.x86.tdpbssd
// CHECK_SPR_SUCCESS: call void @llvm.x86.tdpbsud
// CHECK_SPR_SUCCESS: call void @llvm.x86.tdpbusd
// CHECK_SPR_SUCCESS: call void @llvm.x86.tdpbuud

// CHECK_GNR_SUCCESS: call void @llvm.x86.tdpbssd
// CHECK_GNR_SUCCESS: call void @llvm.x86.tdpbsud
// CHECK_GNR_SUCCESS: call void @llvm.x86.tdpbusd
// CHECK_GNR_SUCCESS: call void @llvm.x86.tdpbuud

// CHECK_DMR_SUCCESS: call void @llvm.x86.tdpbssd
// CHECK_DMR_SUCCESS: call void @llvm.x86.tdpbsud
// CHECK_DMR_SUCCESS: call void @llvm.x86.tdpbusd
// CHECK_DMR_SUCCESS: call void @llvm.x86.tdpbuud

__attribute__((external_only))
export void test_amx_int8() {
    // require amxint8 feature (SPR+)
    @llvm.x86.tdpbssd((uniform int8)2, (uniform int8)0, (uniform int8)1);
    @llvm.x86.tdpbsud((uniform int8)2, (uniform int8)0, (uniform int8)1);
    @llvm.x86.tdpbusd((uniform int8)2, (uniform int8)0, (uniform int8)1);
    @llvm.x86.tdpbuud((uniform int8)2, (uniform int8)0, (uniform int8)1);
}
