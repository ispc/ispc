// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Can't apply "unsigned" qualifier to "varying float" type

struct Bar {
    unsigned float x;
};

// CHECK: Error: "unsigned" qualifier is illegal with "varying float" type

unsigned float foo = 1;

// CHECK: Error: "unsigned" qualifier is illegal with "varying struct Foo" typ

struct Foo {
    float x;
};

unsigned Foo f;