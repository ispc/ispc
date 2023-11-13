// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Illegal to pass a "varying" lvalue to a reference parameter

void inc(float &x) { ++x; }

void foo(uniform float a[], int index) {
    inc(a[index]);
}
