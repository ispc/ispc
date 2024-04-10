// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: "task" qualifier is illegal outside of function declarations

struct Foo {
    task float x;
};

// CHECK: Error: Member operator "." can't be applied to declared but not defined struct type

struct Foo1;

int bar1(Foo1 & foo) { return foo.x; }

// CHECK: Error: Member operator "->" can't be applied to declared but not defined struct type

struct Foo2;

int bar2(Foo2 * uniform foo) { return foo->x; }

// CHECK: Error: Can't compute the size of declared but not defined struct type

struct Foo6;

uniform int bar3() {
    return sizeof(Foo6);
}