// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Must provide initializer for reference-type variable

void func(int a) {
    int &b;
    b = 0;
}
