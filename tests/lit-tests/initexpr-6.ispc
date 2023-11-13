// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Illegal to declare a global variable with unsized array dimensions that aren't set with an initializer expression

int a[][];

