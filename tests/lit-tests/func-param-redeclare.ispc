// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Ignoring redeclaration of symbol "a"

int foo(int a[], int a) {
    return a[i];
}
