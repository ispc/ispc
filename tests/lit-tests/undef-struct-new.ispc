// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Can't dynamically allocate storage for declared but not defined type

struct Foo;

Foo * uniform bar() {
    return uniform new Foo;
}
