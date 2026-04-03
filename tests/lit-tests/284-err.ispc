// Regression test for https://github.com/ispc/ispc/issues/284
// A forward-declared struct can't be used through a pointer before it is
// defined, and a struct with the same name defined in an inner scope is a
// different type.

// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -o - 2>&1 | FileCheck %s --implicit-check-not="Assertion failed"

struct Foo;

struct Bar {
    Foo *foo;
};

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Member operator "->" can't be applied to declared but not defined struct type "uniform struct Foo".
int early(Bar * uniform b) { return b->foo->x; }

int shadow(Bar * uniform b) {
    struct Foo {
        int z;
    };
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Member operator "->" can't be applied to declared but not defined struct type "uniform struct Foo".
    return b->foo->z;
}

struct Foo {
    int x;
};

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Member operator "." can't be applied to pointer type "uniform struct Foo". Did you mean to use "->"?
int dot(Bar * uniform b) { return b->foo.x; }
