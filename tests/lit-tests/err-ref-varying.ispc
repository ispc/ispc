// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Illegal to pass a "varying" lvalue

void foo(float &x) {
    ++x;
}

void bar(uniform float a[], int i) {
    foo(a[i]);
}
