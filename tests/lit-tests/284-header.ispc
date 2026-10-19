// Regression test for https://github.com/ispc/ispc/issues/284
// A struct that is only reachable from exported functions through a pointer
// created while it was forward declared must still be emitted in the header,
// and a pointer to its varying version must name the varying struct. The
// cycle through the self-referential structs must terminate, and the header
// must be usable from C++.  The checks of exported parameters, type export
// lists and the nanobind wrapper must see the definition too, and the checks
// must terminate for a self-referential struct reached through an aligned
// typedef.

// RUN: %{ispc} %s --target=host --nostdlib --nowrap -DISPC -o %t.o -h %t.h --nanobind-wrapper=%t.cpp 2>&1 | FileCheck %s --check-prefix=WARN
// RUN: FileCheck --input-file=%t.h %s
// RUN: FileCheck --input-file=%t.cpp %s --check-prefix=NB
// RUN: %{cxx} -x c++ -fsyntax-only %s --include %t.h

// REQUIRES: !MACOS_HOST

// CHECK-DAG: struct Node {
// CHECK-DAG: struct Bar {
// CHECK-DAG: struct Foo {
// CHECK-DAG: int32_t x;
// CHECK-DAG: struct v{{[0-9]+}}_varying_V * vv;
// CHECK-DAG: int32_t y[{{[0-9]+}}];
// CHECK-DAG: struct Exported {
// CHECK-DAG: struct Aligned {

// WARN: Warning: Exported function parameter "n" points to varying type
// WARN-NOT: Error

// NB: void early(Foo* p) {
// NB-NEXT: ispc::early(p);

#ifdef ISPC
struct Foo;
struct V;
struct Exported;
typedef struct Exported ExportedT;
struct Aligned;
typedef struct Aligned __attribute__((aligned(16))) AlignedT;

struct Bar {
    Foo *foo;
    varying V *uniform vv;
};

struct Node {
    Node *next;
    Bar *bar;
};

// Defined before Foo, so its parameter type refers to the forward
// declaration.
export void early(uniform Foo *uniform p) {}

struct Foo {
    int x;
    Node *node;
};

struct V {
    int y;
};

struct Exported {
    int e;
};

struct Aligned {
    AlignedT *uniform next;
};

export { ExportedT };

export void f(uniform Node *uniform n) { n->bar->foo->x = 1; }
export void aligned(uniform Aligned *uniform a) {}
#else
int g(ispc::Node *n) { return n->bar->foo->x + n->bar->vv->y[0] + n->next->bar->foo->node->bar->foo->x; }
#endif
