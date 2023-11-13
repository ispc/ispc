// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Inconsistent initializer expression list lengths make it impossible

int a[2][] = { { 1, 2, 3 }, { 1, 2, 3, 4 } };

