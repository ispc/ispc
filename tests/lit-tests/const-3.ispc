// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK:  Can't assign to type

struct Foo {
    int x;
};

int func(const Foo f) {
    f.x = 0;
}
