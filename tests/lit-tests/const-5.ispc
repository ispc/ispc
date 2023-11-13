// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Can't assign to type

int func() {
    const int a[10] = {1,2,3,4,5,6,7,8,9,10};
    a[0] = 1;
}
