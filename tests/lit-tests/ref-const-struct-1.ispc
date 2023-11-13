// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Can't assign to type "const varying int32" on left-hand side of expression

struct Foo {
    int x;
};

void f(const Foo &f) {
    f.x += 2;
}
