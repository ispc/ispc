// The test checks that for nested template instantiations function parameters of known type (not templated type) works fine.

// RUN: %{ispc} %s --ast-dump --target=host --nostdlib --nowrap | FileCheck %s -check-prefix=CHECK-AST
// RUN: %{ispc} %s --target=host --nostdlib -o %t.o

// CHECK-AST: FunctionTemplate {{.*}} [ uniform T(uniform int32 I)] "foo1"
// CHECK-AST: (instantiation <uniform int32>) {{.*}}  [ uniform int32(uniform int32 I)] "foo1"
// CHECK-AST: FunctionTemplate {{.*}} [ uniform T(uniform int32 J)] "foo2"
// CHECK-AST: (instantiation <uniform int32>) {{.*}}  [ uniform int32(uniform int32 J)] "foo2"

// CHECK-NOT: FATAL ERROR
template <typename T> uniform T foo1(uniform int I) {
  return I + 1;
}

template <typename T> uniform T foo2(uniform int J) {
  return foo1<T>(J) + 1;
}

uniform int bar() {
  return foo2<uniform int>(1);
}

