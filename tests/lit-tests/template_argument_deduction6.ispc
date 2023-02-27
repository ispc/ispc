// Check array to pointer decay in template argument deduction rules.
// RUN: %{ispc} %s --nostdlib --target=host --ast-dump -o %t.o | FileCheck %s

uniform int foo1(uniform int a[10], uniform int i) {
  return a[i];
}

// CHECK: (instantiation <uniform int32>) Function {{.*}} [ uniform int32(uniform int32 * uniform a, uniform int32 i)] "foo2"
// CHECK: (instantiation <uniform float>) Function {{.*}} [ uniform float(uniform float * uniform a, uniform int32 i)] "foo2"
template <typename T> T foo2(T a[10], uniform int i) {
  return a[i];
}

void bar() {
  uniform int int_array[100];
  uniform float float_array[10];

  // CHECK: FunctionSymbolExpr {{.*}} [ uniform int32(uniform int32 * uniform a, uniform int32 i)] function name: foo1
  foo1(int_array, 5);
  // CHECK: FunctionSymbolExpr {{.*}} [ uniform int32(uniform int32 * uniform a, uniform int32 i)] function name: foo2
  foo2(int_array, 5);
  // CHECK: FunctionSymbolExpr {{.*}} [ uniform float(uniform float * uniform a, uniform int32 i)] function name: foo2
  foo2(float_array, 5);
}
