// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Illegal to take address of non-lvalue or function

void foo() {
    int *ptr = &(1+1);
}
