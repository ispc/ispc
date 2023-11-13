// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Illegal to assign to type "varying struct Foo" due to element "a" with type "const varying int32"

struct Foo {
    const int a;
};

void foo(Foo f) {
    Foo a;
    a = f;
}
