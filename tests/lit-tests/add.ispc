//; RUN: %{ispc} --emit-asm %s -o - | FileCheck %s
float func(float a, float b) {
//; CHECK: vaddps
    return a + b;
}