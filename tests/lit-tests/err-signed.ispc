// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: "signed" qualifier is illegal with non-integer type "varying float"

int foo() {
    signed float x;
}

// CHECK: Error: Can't apply "signed" qualifier to "varying float" type

struct Foo {
    signed float x;
};

// CHECK: Error: "signed" qualifier is illegal with non-integer type "varying struct Foo2"

struct Foo2 {
    float x;
};

signed Foo2 f;

// CHECK: Error: Illegal to apply both "signed" and "unsigned" qualifiers

int foo4() {
    signed unsigned int x;
}