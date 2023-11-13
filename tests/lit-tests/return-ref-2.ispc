// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Illegal to return varying lvalue type from function returning a reference type

float &func(uniform float a[], int i, float f) {
    return a[i];
}
