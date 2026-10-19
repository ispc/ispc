// Regression test for https://github.com/ispc/ispc/issues/284
// A struct that is only used through pointers created while it was forward
// declared must get a complete debug type, not only a forward declaration,
// including when it is self-referential, also through an aligned typedef,
// and when structs with the same name are defined in different functions.

// RUN: %{ispc} %s --target=host --nostdlib --nowrap -g --emit-llvm-text -o - | FileCheck %s

struct Foo;
struct Node;
struct Aligned;
typedef struct Aligned __attribute__((aligned(16))) AlignedT;

struct Bar {
    Foo *foo;
    Node *node;
    AlignedT *uniform aligned;
};

struct Foo {
    int x;
};

struct Node {
    Node *next;
    int v;
};

struct Aligned {
    AlignedT *uniform next;
    int a;
};

export uniform int get(uniform Bar *uniform b) { return b->foo->x + b->node->next->v + b->aligned->next->a; }

uniform int first(void *uniform v) {
    struct L;
    uniform L *uniform p = (uniform L * uniform)v;
    struct L {
        int first;
    };
    return p->first;
}

uniform int second(void *uniform v) {
    struct L;
    uniform L *uniform p = (uniform L * uniform)v;
    struct L {
        int second;
    };
    return p->second;
}

// CHECK-DAG: = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct Foo", {{.*}}size: 32, {{.*}}elements: ![[FOO:[0-9]+]])
// CHECK-DAG: ![[FOO]] = !{![[X:[0-9]+]]}
// CHECK-DAG: ![[X]] = !DIDerivedType(tag: DW_TAG_member, name: "x",
// CHECK-DAG: = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct Node", {{.*}}elements: ![[NODE:[0-9]+]])
// CHECK-DAG: ![[NODE]] = !{![[NEXT:[0-9]+]], ![[V:[0-9]+]]}
// CHECK-DAG: ![[NEXT]] = !DIDerivedType(tag: DW_TAG_member, name: "next",
// CHECK-DAG: ![[V]] = !DIDerivedType(tag: DW_TAG_member, name: "v",
// CHECK-DAG: = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct Aligned", {{.*}}elements: ![[ALIGNED:[0-9]+]])
// CHECK-DAG: ![[ALIGNED]] = !{![[ANEXT:[0-9]+]], ![[A:[0-9]+]]}
// CHECK-DAG: ![[ANEXT]] = !DIDerivedType(tag: DW_TAG_member, name: "next",
// CHECK-DAG: ![[A]] = !DIDerivedType(tag: DW_TAG_member, name: "a",
// CHECK-DAG: = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct L", {{.*}}elements: ![[L1:[0-9]+]])
// CHECK-DAG: ![[L1]] = !{![[FIRST:[0-9]+]]}
// CHECK-DAG: ![[FIRST]] = !DIDerivedType(tag: DW_TAG_member, name: "first",
// CHECK-DAG: = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct L", {{.*}}elements: ![[L2:[0-9]+]])
// CHECK-DAG: ![[L2]] = !{![[SECOND:[0-9]+]]}
// CHECK-DAG: ![[SECOND]] = !DIDerivedType(tag: DW_TAG_member, name: "second",
