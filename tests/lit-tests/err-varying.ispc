// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Can't type cast from type "varying int32" to type "uniform int32"

uniform int foo(int x)  {
    return (uniform int) x;
}

// CHECK: Error: Illegal to pass a "varying" lvalue to a reference parameter

void inc(float &x) { ++x; }

void foo(uniform float a[], int index) {
    inc(a[index]);
}