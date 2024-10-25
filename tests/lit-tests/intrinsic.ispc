// RUN: %{ispc} %s --target=host --emit-llvm-text --enable-llvm-intrinsics -o - | FileCheck %s
// CHECK: declare i16 @llvm.convert.to.fp16.f32(float)

uniform int16 foo(uniform float arg) {
    uniform int16 ret = @llvm.convert.to.fp16.f32(arg);
    return ret;
}
