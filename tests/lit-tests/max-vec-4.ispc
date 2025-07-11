// This test checks error handling when max is called with unsupported arguments

// RUN: not %{ispc} --nowrap --target=host -o %t.o 2>&1 %s | FileCheck %s

#include <short_vec.isph>

// CHECK: Error: Multiple overloaded functions matched call to function "max". 
// CHECK-NEXT: Passed types: (varying bool, varying bool) 
bool<3> foo(bool<3> a, bool<3> b) {
  return max(a, b);
}

struct S { int a; int b; };

// CHECK: Error: Unable to find any matching overload for call to function "max".
// CHECK-NEXT: Passed types: (varying struct S, varying struct S)
S bar(S a, S b) {
  return max(a, b);
}
