// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Illegal to assign to type "varying struct Bar" in type "varying struct Foo" due to element "a" with type "const varying int32"

struct Bar {
    const int a;
};

struct Foo {
    struct Bar b;
};

void foo(Foo f) {
    Foo g;
    g = f;
}
