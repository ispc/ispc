// RUN: %{ispc} %s --cpu=icelake-client --target=avx512skx-i32x8 --emit-llvm-text --enable-llvm-intrinsics -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK: declare <8 x i32> @llvm.x86.avx512.vpdpbusd.256(
int foo_busd(int arg0, int arg1, int arg2) {
    int ret = @llvm.x86.avx512.vpdpbusd.256(arg0, arg1, arg2);
    return ret;
}

// CHECK: declare <8 x i32> @llvm.x86.avx512.vpdpbusds.256(
int foo_busds(int arg0, int arg1, int arg2) {
    int ret = @llvm.x86.avx512.vpdpbusds.256(arg0, arg1, arg2);
    return ret;
}

// CHECK: declare <8 x i32> @llvm.x86.avx512.vpdpwssd.256(
int foo_wssd(int arg0, int arg1, int arg2) {
    int ret = @llvm.x86.avx512.vpdpwssd.256(arg0, arg1, arg2);
    return ret;
}

// CHECK: declare <8 x i32> @llvm.x86.avx512.vpdpwssds.256(
int foo_wssds(int arg0, int arg1, int arg2) {
    int ret = @llvm.x86.avx512.vpdpwssds.256(arg0, arg1, arg2);
    return ret;
}
