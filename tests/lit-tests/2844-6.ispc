// Test for incomplete types regarding unsized arrays.

// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o %t.o 2>&1 | FileCheck %s

// CHECK: Error: incomplete type 'S1'
struct S1 { int a[]; };

// CHECK-NOT: Error: incomplete type 'S2'
struct S2 { int y; int x[]; };

// CHECK-NOT: Error: parameter 'b' has incomplete type
int foo(int b[]);
// CHECK-NOT: Error: parameter 'c' has incomplete type
int boo(int c[]) { return c[0]; }

// CHECK-NOT: FATAL ERROR:
