// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: "unsigned" qualifier is illegal with "varying struct Foo" typ

struct Foo {
    float x;
};

unsigned Foo f;
