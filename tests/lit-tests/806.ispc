// Regression test for https://github.com/ispc/ispc/issues/806
// A store through a member pointer to a forward-declared struct, reached
// through a varying pointer, works once the struct is defined.

// RUN: %{ispc} %s --target=host --nostdlib --nowrap -o %t.o

struct Foo;

struct Bar {
    uniform Foo *uniform foo;
};

struct Foo {
    int i;
};

void bar(Bar *bar) { bar->foo->i = 0; }
