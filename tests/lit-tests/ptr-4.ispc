// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Illegal to perform pointer arithmetic

void *foo(void *p) {
    p += 1;
    return p;
}
