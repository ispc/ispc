// This test checks that AMX BF16 (amxbf16) intrinsics are correctly handled by ISPC

// RUN: not %{ispc} --nowrap %s --target=avx512icl-x4 --enable-llvm-intrinsics --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_ICL_ERROR
// RUN: %{ispc} --nowrap %s --target=avx512spr-x4 --enable-llvm-intrinsics --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_SPR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx512spr-x4 --cpu=gnr --enable-llvm-intrinsics --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_GNR_SUCCESS
// RUN: %{ispc} --nowrap %s --target=avx512spr-x4 --cpu=dmr --enable-llvm-intrinsics --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_DMR_SUCCESS

// REQUIRES: X86_ENABLED && LLVM_20_0+

// CHECK_ICL_ERROR: Target specific LLVM intrinsic "tdpbf16ps" requires feature "amxbf16" not supported on "icelake-client" CPU
// CHECK_SPR_SUCCESS: call void @llvm.x86.tdpbf16ps
// CHECK_GNR_SUCCESS: call void @llvm.x86.tdpbf16ps
// CHECK_DMR_SUCCESS: call void @llvm.x86.tdpbf16ps

__attribute__((external_only))
export void test_amx_bf16() {
    // require amxbf16 feature (SPR+)
    @llvm.x86.tdpbf16ps((uniform int8)2, (uniform int8)0, (uniform int8)1);
}
