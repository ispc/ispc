// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Can't convert between different struct types "varying struct Foo" and "varying struct Bar"

struct Foo {
    int a; 
};

struct Bar {
    int x; 
};

void f(Foo a) { 
    Bar b;
    b  =a;
}
