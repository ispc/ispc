// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: operator "." can't be applied to pointer type

struct Foo { int x; };

int func(Foo *a) {
    a.x = 0;
}
