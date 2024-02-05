// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Initializer for reference-type variable "x" must have an lvalue type

float &func(uniform float a[], int i, float f) {
    float &x = 1.; // a[i];
}

// CHECK: Error: Initializer for reference-type variable "x" must have a uniform lvalue type

float &func2(uniform float a[], int i, float f) {
    float &x = a[i];
}

// CHECK: Error: Initializer for reference-type variable "x" must have a uniform lvalue type

float &func3(uniform int a[], int i, float f) {
    float &x = a[i];
}