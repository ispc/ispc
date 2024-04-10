// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Assignment operator "+=" is illegal with struct type

struct Point { float x, y, z; };

void foo() {
    Point a = {1,2,3}, b = {4,5,6};
    a += b;
}
