// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: "varying" qualifier is illegal with "void" type

varying void *foo;

// CHECK: Error: "uniform" qualifier is illegal with "void" type

struct FFF {
    uniform void *x;
};

// CHECK: Error: "varying" qualifier is illegal with "void" type

varying void *foo() { }

// CHECK: Error: "varying" qualifier is illegal with "void" type

void foo(varying void *x) { }