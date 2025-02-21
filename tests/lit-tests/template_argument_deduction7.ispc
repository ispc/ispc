// Check short vectors in template argument deduction.
// Principal case.

// RUN: %{ispc} %s --nowrap --nostdlib --target=host --ast-dump -o %t.o 2>&1 | FileCheck %s

// CHECK-NOT: Error: Unable to find any matching overload for call to function "foo".

// CHECK: (instantiation <varying int32, 4>) Function {{.*}} [ uniform int32<4>(uniform int32<4> a)] "foo"
template <typename T, uint N> uniform T<N> foo(uniform T<N> a) { return a; }
// CHECK: (func) FunctionSymbolExpr {{.*}} [ uniform int32<4>(uniform int32<4> a)] function name: foo
uniform int<4> boo(uniform int<4> x) { return foo(x); }
