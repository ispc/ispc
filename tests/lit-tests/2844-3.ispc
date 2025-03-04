// Check that we catch incomplete struct types in variable definitions

// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o %t.o 2>&1 | FileCheck %s

// CHECK: Error: variable 'x' has initializer but incomplete struct type
int foo() { struct Sx x = { 0 }; }
// CHECK: Error: variable 'y' has incomplete struct type 'varying struct Sy' and cannot be defined
int bar() { struct Sy y; }

// CHECK: Error: Attempt to allocate memory for incomplete type "varying struct Sz[10]".
int func() { struct Sz z[10]; }
// CHECK-NOT: Error: Attempt to allocate memory for incomplete type "varying struct St[10]".
int func2() { struct St* p[10]; }

// CHECK-NOT: FATAL ERROR:
