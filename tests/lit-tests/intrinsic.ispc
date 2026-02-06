// RUN: %{ispc} %s --target=host --emit-llvm-text --enable-llvm-intrinsics -o - | FileCheck %s
// CHECK: declare float @llvm.fabs.f32(float)

uniform float foo(uniform float arg) {
    uniform float ret = @llvm.fabs.f32(arg);
    return ret;
}
