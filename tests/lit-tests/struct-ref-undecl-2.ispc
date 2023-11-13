// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Member operator "->" can't be applied to declared but not defined struct type

struct Foo;

int bar(Foo * uniform foo) { return foo->x; }
