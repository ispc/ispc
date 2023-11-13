// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Illegal to pre/post increment pointer to undefined struct type

struct Foo;

Foo * uniform bar(Foo * uniform f) {
    return ++f;
}
