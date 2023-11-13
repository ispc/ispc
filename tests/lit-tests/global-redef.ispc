// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Redefinition of variable "foo" is illegal

int foo;
int foo;
