// RUN: not %{ispc} --target=host --nostdlib --nowrap -o - --emit-llvm-text %s 2>&1 | FileCheck %s

// CHECK-NOT: Error: Assertion failed
export struct Matrix {
    float e[16];
};

export void Test(
    uniform Matrix const m[],
    int    i)
{
    m[i] * 1.f +
    m[i] * 1.f +
    m[i];
}
