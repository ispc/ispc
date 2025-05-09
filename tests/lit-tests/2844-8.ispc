// Test that we catch incomplete struct types in global variable declarations.

// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o %t.o 2>&1 | FileCheck %s

// CHECK: Error: variable "x" has incomplete type "varying struct S"
struct S x;
