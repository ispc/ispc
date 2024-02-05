// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Array dimension must be representable with a 32-bit integer.

struct foo {
    int x[0xffffffffffff];
};

// CHECK: Error: Array dimension must be non-negative
struct boo {
    int x[-1];
};

// CHECK: Error: Can only pre/post increment numeric and pointer types

void coo() {
    float a[5] = { 1,2,3,4,5};
    ++a;
}

// CHECK: Error: Illegal to assign to array type "varying float[5]"

void doo(float *x) {
    float a[5] = { 1,2,3,4,5};
    a += 3;
}

// CHECK: Error: Illegal to assign to array type "varying float[3]"

void eoo(float *x) {
    float a[3] = { 1,2,3 };
    a = x;
}
