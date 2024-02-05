// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Ignoring redeclaration of symbol "a"

void foo() {
    int a;
    int a;
}
