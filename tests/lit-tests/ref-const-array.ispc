// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Can't assign to type "const varying int32" on left-hand side of expression

const int x[20];

void foo() {
    ++x[5];
}
