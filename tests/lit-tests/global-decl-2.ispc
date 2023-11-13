// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Definition of variable "foo" conflicts with definition at 

extern int foo;
extern float foo;
