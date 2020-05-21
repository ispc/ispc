//; RUN: %{ispc} --emit-asm %s --target=avx2-i32x8 --arch=x86-64 -o - | FileCheck %s
//; REQUIRES: X86_ENABLED
float func(float a, float b) {
//; CHECK: vaddps
    return a + b;
}
