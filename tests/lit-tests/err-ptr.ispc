// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Illegal to pre/post increment

void *foo1(void *p) {
    ++p;
}

// CHECK: Error: Illegal to perform pointer arithmetic

void *foo3(void *p) {
    p = p-2;
    return p;
}

// CHECK: Error: Illegal to perform pointer arithmetic

void *foo4(void *p) {
    p += 1;
    return p;
}

// CHECK: Error: Can't assign to type "const uniform int32 * const varying"

void fooptr1(const int * const p) {
    ++p;
}

// CHECK: Error: Can't assign to type "const varying int32" on left-hand side

void fooptr2(const int * p) {
    *p = 0;
}

// CHECK: Error: Conversion between incompatible pointer types

export void f_f_in(uniform float RET[], uniform float aFOO[]) {
    uniform int8 * varying pa = (uniform int8 *)aFOO;
    RET[programIndex] = aFOO - pa;
}

export void result_in(uniform float RET[]) {
    RET[programIndex] = 40;
}

// CHECK: Error: Can't convert from pointer type "void * varying" to incompatible pointer type "uniform int32 * varying" for return statement

int *foo(void *p) {
    return p;
}