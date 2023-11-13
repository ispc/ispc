// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Can't assign to type "const uniform int32" on left-hand side of expression

void foo(const uniform int &x) {
    x = 0;
}
