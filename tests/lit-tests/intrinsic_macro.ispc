// RUN: %{ispc} %s --target=host --nowrap -O2 --emit-llvm-text -o - | FileCheck %s -check-prefix=CHECK_DISABLED
// RUN: %{ispc} %s --target=host --nowrap -O2 --emit-llvm-text --enable-llvm-intrinsics -o - | FileCheck %s -check-prefix=CHECK_ENABLED

// REQUIRES: X86_ENABLED

// CHECK_DISABLED-NOT: call float @llvm.sqrt.f32(
// CHECK_ENABLED: call float @llvm.sqrt.f32(
uniform float foo(uniform float a, uniform int b) {
    uniform float ret = -1;
#ifdef ISPC_LLVM_INTRINSICS_ENABLED
    ret = @llvm.sqrt.f32(a);
#endif
    return ret;
}

