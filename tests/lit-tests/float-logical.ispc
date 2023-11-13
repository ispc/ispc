// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Illegal to use ^= operator with floating-point

float foo(float a, float b) {
    return a ^= b;
}
