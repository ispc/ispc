// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Gather operation is impossible due to the presence of struct member "x" with uniform type

struct Bar { uniform int x; };
struct Foo { varying Bar b; };

void a(uniform Foo * varying array, int index) {
    Foo v = array[index];
}
