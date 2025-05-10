// The test checks that for nested template instantiations without function parameters works fine.

// RUN: %{ispc} %s --ast-dump --target=host --nostdlib --nowrap | FileCheck %s -check-prefix=CHECK-AST
// RUN: %{ispc} %s --target=host --nostdlib -o %t.o

// CHECK-AST: FunctionTemplate {{.*}} [ uniform T()] "foo1"
// CHECK-AST: (instantiation <varying int32>) {{.*}}  [ uniform int32()] "foo1"
// CHECK-AST: FunctionTemplate {{.*}} [ uniform T()] "foo2"
// CHECK-AST: (instantiation <varying int32>) {{.*}}  [ uniform int32()] "foo2"

// CHECK-NOT: FATAL ERROR
template <typename T> uniform T foo1() {
  return programCount + 1;
}

template <typename T> uniform T foo2() {
  return foo1<T>() + 1;
}

uniform int bar() {
  return foo2<int>();
}

