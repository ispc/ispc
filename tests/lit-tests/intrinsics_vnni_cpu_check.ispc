// Test that avx512vnni intrinsics work on ICL (which has avx512_vnni) but not on SKX
// RUN: %{ispc} %s --cpu=icelake-client --target=avx512skx-x8 --emit-llvm-text --enable-llvm-intrinsics -o - | FileCheck %s --check-prefix=CHECK-ICL
// RUN: not %{ispc} %s --cpu=skx --target=avx512skx-x8 --emit-llvm-text --enable-llvm-intrinsics -o - 2>&1 | FileCheck %s --check-prefix=CHECK-SKX

// REQUIRES: X86_ENABLED
// UNSUPPORTED: LLVM_22_0+

// CHECK-ICL: declare <8 x i32> @llvm.x86.avx512.vpdpbusd.256(
// CHECK-SKX: requires feature "avx512_vnni" not supported on "skx" CPU
int foo_busd(int arg0, int arg1, int arg2) {
    int ret = @llvm.x86.avx512.vpdpbusd.256(arg0, arg1, arg2);
    return ret;
}
