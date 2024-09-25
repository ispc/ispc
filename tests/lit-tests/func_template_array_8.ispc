// Check compiler diagnostics when arrays are declared incorrectly inside templates.
// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Illegal to assign to array type "varying float[5]"
template <typename T, uint C>
void doo(uniform T *varying x) {
    T a[C] = { 1,2,3,4,5};
    a += 3;
}

void test(float *x) {
    doo<float, 5>(x);
}

// CHECK: Error: Arrays with unsized dimensions in dimensions after the first one are illegal in function parameter lists
template <uint C>
void func(int a[C][]) {
}