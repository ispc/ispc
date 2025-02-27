// Test that we catch incomplete struct types.

// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o %t.o 2>&1 | FileCheck %s

// CHECK: Error: incomplete type 'S1'.
// CHECK-NEXT: - field 'a' has incomplete type '/*unbound*/ struct S2'.
struct S1 { struct S2 a; };

// CHECK-NOT: Error: incomplete type 'C1'. 
// CHECK-NOT: - field 'c' has incomplete type
struct C1 { struct I1 *c; };
