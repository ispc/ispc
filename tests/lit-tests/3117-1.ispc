// RUN: %{ispc} --target=host --nowrap --enable-llvm-intrinsics --emit-llvm-text %s -o - 2>&1 | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-DAG: %calltmp = tail call float @llvm.maxnum.f32(float %a, float %b)

// CHECK-NOT: Intrinsic name not mangled correctly for type arguments! Should be: llvm.maxnum.f32
uniform float foo(uniform float a, uniform float b) {
    return @llvm.maxnum(a, b);
}
