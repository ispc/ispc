// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Illegal to apply both "signed" and "unsigned" qualifiers

int foo() {
    signed unsigned int x;
}
