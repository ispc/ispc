// Regression test for https://github.com/ispc/ispc/issues/284
// A struct that is only reachable from exported functions through a pointer
// created while it was forward declared must still be emitted in the header.
// The cycle through the self-referential structs must terminate.

// RUN: %{ispc} %s --target=host --nostdlib --nowrap -o %t.o -h %t.h
// RUN: FileCheck --input-file=%t.h %s

struct Foo;

struct Bar {
    Foo *foo;
};

struct Node {
    Node *next;
    Bar *bar;
};

struct Foo {
    int x;
    Node *node;
};

// CHECK-DAG: struct Node {
// CHECK-DAG: struct Bar {
// CHECK-DAG: struct Foo {
// CHECK-DAG: int32_t x;
export void f(uniform Node *uniform n) { n->bar->foo->x = 1; }
