// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Can only pre/post increment numeric and pointer types

void foo() {
    float a[5] = { 1,2,3,4,5};
    ++a;
}
