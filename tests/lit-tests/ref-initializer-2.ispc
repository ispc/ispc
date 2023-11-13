// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Initializer for reference-type variable "x" must have a uniform lvalue type

float &func(uniform float a[], int i, float f) {
    float &x = a[i];
}

