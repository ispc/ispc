// Calling LLVM intrinsics is enabled by default and ISPC_LLVM_INTRINSICS_ENABLED
// is always defined. --enable-llvm-intrinsics is accepted as a no-op.
// RUN: %{ispc} %s --target=host --nowrap -O2 --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=host --nowrap -O2 --emit-llvm-text --enable-llvm-intrinsics -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK: call float @llvm.sqrt.f32(
uniform float foo(uniform float a, uniform int b) {
    uniform float ret = -1;
#ifdef ISPC_LLVM_INTRINSICS_ENABLED
    ret = @llvm.sqrt.f32(a);
#endif
    return ret;
}
