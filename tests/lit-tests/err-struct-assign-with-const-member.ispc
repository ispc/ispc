// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Illegal to assign to type "varying struct Bar" in type "varying struct Foo" due to element "a" with type "const varying int32"

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

// CHECK: Error: Illegal to assign to type "varying struct Str" due to element "a" with type "const varying int32"

struct Str {
    const int a;
};

void func(Str f) {
    Str a;
    a = f;
}