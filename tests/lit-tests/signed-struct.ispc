// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: "signed" qualifier is illegal with non-integer type "varying struct Foo"

struct Foo {
    float x;
};

signed Foo f;
