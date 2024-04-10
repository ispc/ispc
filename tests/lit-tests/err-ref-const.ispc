// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Can't assign to type "const varying int32" on left-hand side of expression

const int x[20];

void foo1() {
    ++x[5];
}

// CHECK: Error: Can't assign to type "const varying int16" on left-hand side of expression

struct Foo1 {
    int16 x;
};

void f(const Foo1 &f) {
    f.x += 2;
}

// CHECK: Error: Can't assign to type "const varying int8" on left-hand side of expression

struct Foo2 {
    int8 y;
};

void f(const Foo2 &f) {
    ++f.y;
}

// CHECK: Error: Can't assign to type "const uniform int64" on left-hand side of expression

void foo(const uniform int64 &x) {
    x = 0;
}