// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 --enable-llvm-intrinsics -h %t.h --emit-llvm-text -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s

// REQUIRES: XE_ENABLED

// CHECK: call float @llvm.genx.rndu.f32(float %inVal)
export void foo(uniform float inVal, uniform float outArr[]) {
    uniform float outVal = @llvm.genx.rndu.f32(inVal);
    outArr[0] = outVal;
}

