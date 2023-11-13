// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: First operand to binary operator "&" must be an integer or bool

float foo(float a, float b) {
    return a & b;
}
