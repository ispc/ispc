// Test that avx512vnni intrinsics are properly rejected on SKX target
// RUN: not %{ispc} %s --cpu=skx --target=avx512skx-x8 --emit-llvm-text --enable-llvm-intrinsics -o - 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED
// UNSUPPORTED: LLVM_22_0+

// CHECK: requires feature "avx512_vnni" not supported on "skx" CPU
int foo_busd(int arg0, int arg1, int arg2) {
    int ret = @llvm.x86.avx512.vpdpbusd.256(arg0, arg1, arg2);
    return ret;
}

// CHECK: requires feature "avx512_vnni" not supported on "skx" CPU
int foo_busds(int arg0, int arg1, int arg2) {
    int ret = @llvm.x86.avx512.vpdpbusds.256(arg0, arg1, arg2);
    return ret;
}

// CHECK: requires feature "avx512_vnni" not supported on "skx" CPU
int foo_wssd(int arg0, int arg1, int arg2) {
    int ret = @llvm.x86.avx512.vpdpwssd.256(arg0, arg1, arg2);
    return ret;
}

// CHECK: requires feature "avx512_vnni" not supported on "skx" CPU
int foo_wssds(int arg0, int arg1, int arg2) {
    int ret = @llvm.x86.avx512.vpdpwssds.256(arg0, arg1, arg2);
    return ret;
}
