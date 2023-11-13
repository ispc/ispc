// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Can't apply "unsigned" qualifier to "varying float" type

struct Foo {
    unsigned float x;
};
