// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Struct member "a" has same name as a previously-declared member

struct Foo {
    int a, a;
};
