// This test checks that AMX FP8 (amxfp8) intrinsics are correctly handled by ISPC

// RUN: not %{ispc} --nowrap %s --target=avx512spr-x4 --enable-llvm-intrinsics --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_SPR_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx512spr-x4 --cpu=gnr --enable-llvm-intrinsics --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_GNR_ERROR
// RUN: %{ispc} --nowrap %s --target=avx512spr-x4 --cpu=dmr --enable-llvm-intrinsics --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_DMR_SUCCESS

// REQUIRES: X86_ENABLED && LLVM_20_0+

// CHECK_SPR_ERROR: Target specific LLVM intrinsic "tdpbf8ps" requires feature "amxfp8" not supported on "sapphirerapids" CPU
// CHECK_GNR_ERROR: Target specific LLVM intrinsic "tdpbf8ps" requires feature "amxfp8" not supported on "graniterapids" CPU
// CHECK_DMR_SUCCESS: call void @llvm.x86.tdpbf8ps
// CHECK_DMR_SUCCESS: call void @llvm.x86.tdpbhf8ps
// CHECK_DMR_SUCCESS: call void @llvm.x86.tdphbf8ps
// CHECK_DMR_SUCCESS: call void @llvm.x86.tdphf8ps

__attribute__((external_only))
export void test_amx_fp8() {
    // require amxfp8 feature (DMR+)
    @llvm.x86.tdpbf8ps((uniform int8)2, (uniform int8)0, (uniform int8)1);
    @llvm.x86.tdpbhf8ps((uniform int8)2, (uniform int8)0, (uniform int8)1);
    @llvm.x86.tdphbf8ps((uniform int8)2, (uniform int8)0, (uniform int8)1);
    @llvm.x86.tdphf8ps((uniform int8)2, (uniform int8)0, (uniform int8)1);
}
