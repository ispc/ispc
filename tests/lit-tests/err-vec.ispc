// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Can't convert between differently sized vector types

void foo(float<3> a, int<2> b) {
    a += b;
}
