// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Can't convert from pointer type "void * varying" to incompatible pointer type "uniform int32 * varying" for return statement

int *foo(void *p) {
    return p;
}
