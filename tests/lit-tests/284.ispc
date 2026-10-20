// Regression test for https://github.com/ispc/ispc/issues/284
// Types created from a forward-declared struct before it is defined (pointers
// and references to it) must see the definition once it is provided.

// RUN: mkdir -p %t.dir
// RUN: echo "struct I { int x; };" > %t.dir/284-local-def.isph
// RUN: %{ispc} %s -I %t.dir --target=host --nostdlib --nowrap --emit-llvm-text -o - 2>&1 | FileCheck %s --implicit-check-not="Assertion failed"

struct Foo;

struct Bar {
    Foo *foo;
};

uniform Foo * uniform gp;
uniform Foo &getRef();
Foo getVal();
uniform Foo getUVal();
void takeConstRef(const uniform Foo &r);
typedef struct Foo FooT;
typedef Foo (*FooFn)();

// Templates see the definition at the point of instantiation.
template <typename T> uniform int early(T *uniform b) { return b->foo->x + sizeof(uniform Foo); }
template <typename T> uniform int earlyIndex(T *uniform b) { return b->foo[1].x; }
// Typedefs keep their uniform/varying qualifiers.
typedef uniform struct Foo *uniform FooP;
typedef uniform struct Foo FooU;
typedef varying struct Foo FooV;
FooP gq;

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

// Values returned by functions declared before the definition, which are
// stored to temporaries.
// CHECK-LABEL: @byValue___
int byValue(bool c, Foo v) {
    Foo s = c ? getVal() : v;
    takeConstRef(getUVal());
    return getVal().x + s.x;
}

// A varying function pointer whose type was declared before the definition:
// the per-instance calls accumulate their results in a temporary.
// CHECK-LABEL: @viaVaryingFnPtr___
int viaVaryingFnPtr(FooFn f) { return f().x; }

template <typename T> uniform int value(T *uniform p) { return p->x; }
template <typename T> uniform int valueRef(T &r) { return r.x; }

// CHECK-LABEL: @instantiate___
uniform int instantiate(uniform Bar *uniform b) { return early(b) + earlyIndex(b); }

// Operator overloads are found through references created before the
// definition.
uniform Foo operator+(uniform Foo &a, uniform Foo &b);
// CHECK-LABEL: @overload___
uniform int overload() { return (getRef() + getRef()).x; }

// CHECK-LABEL: @deduce___
uniform int deduce(uniform Bar *uniform b) { return value(b->foo) + valueRef(getRef()); }

// CHECK-LABEL: @typedefed___
uniform int typedefed() {
    uniform FooT f;
    f.x = 1;
    return f.x + sizeof(FooT);
}

// The qualifiers of the typedefs must be kept: these return uniform values,
// which is an error if FooP or FooU become varying.
// CHECK-LABEL: @typedefQualifiers___
uniform int typedefQualifiers(FooU &r) {
    FooP q = gq;
    FooU l = r;
    return q->x + l.x;
}

// The varying qualifier of FooV must be kept too. Check it on a pointer
// target, which is uniform by default: sizeof(*p) is the size of the
// varying struct only if *p is varying. Compare against the size of the
// varying struct rather than a constant, as the width of --target=host
// differs between machines.
// CHECK-LABEL: @typedefVarying___
// CHECK: ret i32 1
uniform int typedefVarying(FooV *uniform p) { return sizeof(*p) == sizeof(varying struct Foo) ? 1 : 0; }

// A struct defined in a function body can be used through a pointer created
// before its definition, after the definition.
// CHECK-LABEL: @localAfter___
uniform int localAfter(void *uniform v) {
    struct L;
    uniform L *uniform p = (uniform L * uniform)v;
    struct L {
        int x;
    };
    p += 1;
    void *uniform q = (void *uniform)&p[1];
    return sizeof(uniform L) + (p + 1)->x + (q == v ? 1 : 0);
}

// Self-referential struct: the "next" member is declared while Node is
// still incomplete.
struct Node {
    Node *next;
    int v;
};

// CHECK-LABEL: @node___
int node(Node * uniform n) { return n->next->v + (n->next + 1)->v; }

// A struct defined in a function body by an included file or under a #line
// directive can be used after the definition.
// CHECK-LABEL: @localInclude___
uniform int localInclude(void *uniform v) {
    struct I;
    uniform I *uniform p = (uniform I * uniform)v;
#include "284-local-def.isph"
    return p->x;
}

// CHECK-LABEL: @localLine___
uniform int localLine(void *uniform v) {
    struct N;
    uniform N *uniform p = (uniform N * uniform)v;
#line 1 "other.ispc"
    struct N {
        int x;
    };
#line 1 "third.ispc"
    return p->x;
}

// A template instantiated after a struct defined in a function body sees the
// definition, although the template body comes earlier in the file.
template <typename T> uniform int nextX(T *uniform p) { return p->next->x; }

// CHECK-LABEL: @localTemplate___
uniform int localTemplate(void *uniform v) {
    struct LNode {
        LNode *next;
        int x;
    };
    uniform LNode *uniform p = (uniform LNode * uniform)v;
    return nextX(p);
}

// Like in C++, a template sees a struct defined in the function body that
// instantiates it even if the call comes before the definition.
template <typename T> uniform int pointeeSize(T p) { return sizeof(*p); }

// CHECK-LABEL: @templateBefore___
// CHECK: ret i32 8
uniform int templateBefore(void *uniform v) {
    struct TB;
    uniform TB *uniform p = (uniform TB * uniform)v;
    uniform int n = pointeeSize(p);
    struct TB {
        int x;
        int y;
    };
    return n;
}
