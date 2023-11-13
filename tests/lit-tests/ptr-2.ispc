// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Illegal to pre/post increment

void *foo(void *p) {
    ++p;
}
