// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: "void" type illegal for struct member

struct Foo {
    void x;
};
