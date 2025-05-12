// Check that we catch incomplete struct types in function parameters

// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o %t.o 2>&1 | FileCheck %s

// CHECK: Error: parameter 'a' has incomplete type 'varying struct Sy'
int bar(struct Sy a) {}

// CHECK: Error: parameter 'b' has incomplete type 'varying struct Sx'
int foo(struct Sx b);

// CHECK-NOT: Error: parameter 'n' is an incomplete
int func1(struct Sn *n) {}
// CHECK-NOT: Error: parameter 'm' is an incomplete
int func2(struct Sm &m) {}
// CHECK-NOT: Error: parameter 'q' is an incomplete
int boo(struct Cs q[10]) {}

// CHECK-NOT: FATAL ERROR:
