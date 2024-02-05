// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Arrays with unsized dimensions in dimensions after the first one are illegal in function parameter lists

void func(int a[1][]) {
}
