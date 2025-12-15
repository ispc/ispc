// This test checks that AMX FP16 (amxfp16) intrinsics are correctly handled by ISPC

// RUN: not %{ispc} --nowrap %s --target=avx512spr-x4 --enable-llvm-intrinsics --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_SPR_ERROR
// RUN: %{ispc} --nowrap %s --target=avx512gnr-x4 --enable-llvm-intrinsics --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_GNR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx10.2dmr-x4 --enable-llvm-intrinsics --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_DMR_SUCCESS

// REQUIRES: X86_ENABLED && LLVM_20_0+ && !MACOS_HOST

// CHECK_SPR_ERROR: Target specific LLVM intrinsic "tdpfp16ps" requires feature "amxfp16" not supported on "sapphirerapids" CPU
// CHECK_GNR_SUCCESS: call void @llvm.x86.tdpfp16ps
// CHECK_DMR_SUCCESS: call void @llvm.x86.tdpfp16ps

__attribute__((external_only))
export void test_amx_fp16() {
    // require amxfp16 feature (GNR+)
    @llvm.x86.tdpfp16ps((uniform int8)2, (uniform int8)0, (uniform int8)1);
}
