// Regression test for https://github.com/ispc/ispc/issues/284
// Forward-declared struct should be usable through pointer after full definition.

// RUN: %{ispc} %s --target=host --nostdlib --nowrap --emit-llvm-text -o - 2>&1 | FileCheck %s

struct Foo;

struct Bar {
    Foo *foo;
};

struct Foo {
    int x;
};

// CHECK-NOT: Error: Member operator "->" can't be applied to declared but not defined
// CHECK-LABEL: @func___
// CHECK: ret <[[WIDTH:.*]] x i32>
int func(Bar * uniform b) {
    return b->foo->x;
}
