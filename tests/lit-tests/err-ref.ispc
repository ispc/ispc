// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: syntax error, unexpected '&'

int foo(int & & bar) {
    bar = 0;
    return bar;
}

// CHECK: Error: syntax error, unexpected '*',

void foo3(int & * x) {
    *x = NULL;
}

// CHECK: Error: Must provide initializer for reference-type variable

void func2(int a) {
    int &b;
    b = 0;
}