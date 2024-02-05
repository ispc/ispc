// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Illegal to use ^= operator with floating-point

float foo(float a, float b) {
    return a ^= b;
}

// CHECK: Error: First operand to binary operator "&" must be an integer or bool

float bar(float a, float b) {
    return a & b;
}