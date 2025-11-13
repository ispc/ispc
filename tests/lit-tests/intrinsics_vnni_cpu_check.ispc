// Test that VNNI intrinsics work on ICL (avx512_vnni) and AVX2VNNI targets (avx_vnni), but not on SKX
// Also test that 512-bit VNNI intrinsics are rejected on AVX2-VNNI (which lacks AVX-512)
// RUN: %{ispc} %s --cpu=icelake-client --target=avx512skx-x8 --emit-llvm-text --enable-llvm-intrinsics --nowrap -o - | FileCheck %s --check-prefix=CHECK-ICL
// RUN: %{ispc} %s --target=avx2vnni-i32x8 --emit-llvm-text --enable-llvm-intrinsics --nowrap -o - | FileCheck %s --check-prefix=CHECK-AVX2VNNI
// RUN: not %{ispc} %s --cpu=skx --target=avx512skx-x8 --enable-llvm-intrinsics --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-SKX
// RUN: not %{ispc} %s --target=avx2vnni-i32x8 -DTEST_512BIT --emit-llvm-text --enable-llvm-intrinsics --nowrap -o - 2>&1 | FileCheck %s --check-prefix=CHECK-AVX2VNNI-512

// REQUIRES: X86_ENABLED
// UNSUPPORTED: LLVM_22_0+

// CHECK-ICL: declare <8 x i32> @llvm.x86.avx512.vpdpbusd.256(
// CHECK-AVX2VNNI: declare <8 x i32> @llvm.x86.avx512.vpdpbusd.256(
// CHECK-SKX: feature "avx512_vnni" or "avx_vnni" not supported on "skx" CPU
int foo_busd(int arg0, int arg1, int arg2) {
    int ret = @llvm.x86.avx512.vpdpbusd.256(arg0, arg1, arg2);
    return ret;
}

// CHECK-ICL: declare <8 x i32> @llvm.x86.avx512.vpdpbusds.256(
// CHECK-AVX2VNNI: declare <8 x i32> @llvm.x86.avx512.vpdpbusds.256(
// CHECK-SKX: feature "avx512_vnni" or "avx_vnni" not supported on "skx" CPU
int foo_busds(int arg0, int arg1, int arg2) {
    int ret = @llvm.x86.avx512.vpdpbusds.256(arg0, arg1, arg2);
    return ret;
}

// CHECK-ICL: declare <8 x i32> @llvm.x86.avx512.vpdpwssd.256(
// CHECK-AVX2VNNI: declare <8 x i32> @llvm.x86.avx512.vpdpwssd.256(
// CHECK-SKX: feature "avx512_vnni" or "avx_vnni" not supported on "skx" CPU
int foo_wssd(int arg0, int arg1, int arg2) {
    int ret = @llvm.x86.avx512.vpdpwssd.256(arg0, arg1, arg2);
    return ret;
}

// CHECK-ICL: declare <8 x i32> @llvm.x86.avx512.vpdpwssds.256(
// CHECK-AVX2VNNI: declare <8 x i32> @llvm.x86.avx512.vpdpwssds.256(
// CHECK-SKX: feature "avx512_vnni" or "avx_vnni" not supported on "skx" CPU
int foo_wssds(int arg0, int arg1, int arg2) {
    int ret = @llvm.x86.avx512.vpdpwssds.256(arg0, arg1, arg2);
    return ret;
}

#ifdef TEST_512BIT
// Test that 512-bit VNNI intrinsics are rejected on AVX2-VNNI targets (no AVX-512)
// CHECK-AVX2VNNI-512: requires 512-bit AVX-512 support not available on "alderlake" CPU
uniform int<16> foo_512(uniform int<16> arg0, uniform int<16> arg1, uniform int<16> arg2) {
    uniform int<16> ret = @llvm.x86.avx512.vpdpbusd.512(arg0, arg1, arg2);
    return ret;
}
#endif
