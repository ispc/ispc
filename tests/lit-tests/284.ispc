// Regression test for https://github.com/ispc/ispc/issues/284
// Types created from a forward-declared struct before it is defined (pointers
// and references to it) must see the definition once it is provided.

// RUN: %{ispc} %s --target=host --nostdlib --nowrap --emit-llvm-text -o - 2>&1 | FileCheck %s

// CHECK-NOT: Error
// CHECK-NOT: Assertion failed

struct Foo;

struct Bar {
    Foo *foo;
};

uniform Foo * uniform gp;
uniform Foo &getRef();
typedef struct Foo FooT;

struct Foo {
    int x;
};

// CHECK-LABEL: @member___
int member(Bar * uniform b) { return b->foo->x; }

// CHECK-LABEL: @deref___
int deref(Bar * uniform b) {
    Foo f = *b->foo;
    return f.x + (*b->foo).x + b->foo[1].x;
}

// CHECK-LABEL: @arith___
int arith(Bar * uniform b) { return (b->foo + 1)->x; }

// CHECK-LABEL: @size___
uniform int size(Bar * uniform b) { return sizeof(*b->foo); }

// CHECK-LABEL: @global___
uniform int global() { return gp->x; }

// CHECK-LABEL: @ref___
uniform int ref() { return getRef().x + sizeof(getRef()); }

template <typename T> uniform int value(T *uniform p) { return p->x; }
template <typename T> uniform int valueRef(T &r) { return r.x; }

// CHECK-LABEL: @deduce___
uniform int deduce(uniform Bar *uniform b) { return value(b->foo) + valueRef(getRef()); }

// CHECK-LABEL: @typedefed___
uniform int typedefed() {
    uniform FooT f;
    f.x = 1;
    return f.x + sizeof(FooT);
}

// Self-referential struct: the "next" member is declared while Node is
// still incomplete.
struct Node {
    Node *next;
    int v;
};

// CHECK-LABEL: @node___
int node(Node * uniform n) { return n->next->v + (n->next + 1)->v; }
