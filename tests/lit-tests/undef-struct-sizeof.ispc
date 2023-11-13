// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Can't compute the size of declared but not defined struct type

struct Foo;

uniform int bar() {
    return sizeof(Foo);
}
