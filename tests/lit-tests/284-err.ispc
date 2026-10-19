// Regression test for https://github.com/ispc/ispc/issues/284
// A forward-declared struct can't be used through a pointer before it is
// defined, and a struct with the same name defined in an inner scope is a
// different type.  This holds for uses earlier in the same function body,
// which are type checked after a later definition in the body is parsed, and
// for copies of the whole struct, which are only generated after the struct
// is defined later in the file.

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

void copy(uniform Foo *uniform p, uniform Foo *uniform q) {
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Illegal to perform pointer arithmetic on undefined struct type "uniform struct Foo * uniform".
    p += 1;
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Can't assign to declared but not defined struct type "uniform struct Foo".
    *p = *q;
    // CHECK: :[[@LINE+2]]:5: Error: Illegal to index into pointer to undefined struct type "uniform struct Foo * uniform".
    // CHECK: :[[@LINE+1]]:12: Error: Illegal to index into pointer to undefined struct type "uniform struct Foo * uniform".
    p[0] = q[0];
}

void *uniform address(uniform Foo *uniform p) {
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Illegal to index into pointer to undefined struct type "uniform struct Foo * uniform".
    return (void *uniform)&p[1];
}

struct Foo {
    int x;
};

uniform int local(void *uniform v) {
    struct L;
    uniform L *uniform p = (uniform L * uniform)v;
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Can't compute the size of declared but not defined struct type "uniform struct L".
    uniform int n = sizeof(uniform L);
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Illegal to perform pointer arithmetic on undefined struct type "uniform struct L * uniform".
    p = p + 1;
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Illegal to perform pointer arithmetic on undefined struct type "uniform struct L * uniform".
    n += (uniform int)(p - p);
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Illegal to pre/post increment pointer to undefined struct type "uniform struct L * uniform".
    p++;
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Can't assign to declared but not defined struct type "uniform struct L".
    *p = *p;
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Illegal to perform pointer arithmetic on undefined struct type "uniform struct L * uniform".
    p -= 1;
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Illegal to index into pointer to undefined struct type "uniform struct L * uniform".
    void *uniform q = (void *uniform)&p[1];
    struct L {
        int x;
    };
    return n;
}

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Member operator "." can't be applied to pointer type "uniform struct Foo". Did you mean to use "->"?
int dot(Bar * uniform b) { return b->foo.x; }

uniform int otherFile() {
    struct M;
// Positions in different files can't be ordered, so the definition isn't used.
#line 1 "other.ispc"
    // CHECK: other.ispc:2:{{[0-9]+}}: Error: Can't compute the size of declared but not defined struct type "uniform struct M".
    uniform int n = sizeof(uniform M);
#line 1 "third.ispc"
    struct M {
        int x;
    };
    return n;
}
