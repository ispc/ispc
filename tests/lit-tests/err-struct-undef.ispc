// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

struct Foo;

Foo * uniform bar(uniform int x, Foo * uniform f) {
    if (x) {
        // CHECK: Error: Can't dynamically allocate storage for declared but not defined type
        return uniform new Foo;
    } else {
        // CHECK: Error: Illegal to perform pointer arithmetic on undefined struct type
        return f + 1;
    }
    if (x + 1) {
        // CHECK: Error: Illegal to perform pointer arithmetic on undefined struct type
        return 1 + f;
    } else {
        // CHECK: Error: Illegal to perform pointer arithmetic on undefined struct type
        return f-1;
    }
    // CHECK: Error: Illegal to pre/post increment pointer to undefined struct type
    return ++f;
}