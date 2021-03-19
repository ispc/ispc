// RUN: %{ispc} %s --target=genx-x16 --arch=genx64 --enable-llvm-intrinsics -h %t.h --emit-llvm-text -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s

// REQUIRES: GENX_ENABLED

// CHECK: call float @llvm.genx.rndu.f32(float %inVal)
export void foo(uniform float inVal, uniform float outArr[]) {
    uniform float outVal = @llvm.genx.rndu.f32(inVal);
    outArr[0] = outVal;
}

