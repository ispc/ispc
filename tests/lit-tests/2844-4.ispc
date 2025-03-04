// Test that incomplete struct types are properly diagnosed in initializers and
// variable declarations with recursive struct types with incomplete struct
// types.

// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o %t.o 2>&1 | FileCheck %s

// CHECK: Error: incomplete type 'S1'.
// CHECK-NEXT: - field 'a' has incomplete type '/*unbound*/ struct S2'.
struct S1 { struct S2 a; };
struct S1 func() {
// CHECK: Error: variable 'x' has initializer but incomplete struct type
    struct S1 x = { 0 };
    return x;
}

struct S3 { struct S4 a; };
struct S3 foo() {
// CHECK: Error: variable 'y' has incomplete struct type 'varying struct S3' and cannot be defined
    struct S3 y;
// CHECK: Error: Attempt to allocate memory for incomplete type "varying struct S3".
// CHECK: Error: Malformed return value in function with non-void return type.
    return y;
}

// CHECK-NOT: FATAL ERROR
