// Check how pointer types are instantiated with [T = uniform int] and [T = varying int]
// RUN: %{ispc} %s --target=host --nostdlib --ast-dump -o %t.o | FileCheck %s

// CHECK: (instantiation <varying int32>) Function {{.*}} [ void(varying int32 * varying p1, varying int32 * uniform * varying p2, varying int32 * uniform * uniform * varying p3)] "foo"
// CHECK: (instantiation <uniform int32>) Function {{.*}} [ void(uniform int32 * varying p1, uniform int32 * uniform * varying p2, uniform int32 * uniform * uniform * varying p3)] "foo"
template <typename T> void foo(T *p1, T **p2, T ***p3) { }

void bar(uniform int *up1, uniform int **up2, uniform int ***up3,
         varying int *vp1, varying int **vp2, varying int ***vp3) {
  // CHECK: FunctionSymbolExpr {{.*}} [ void(varying int32 * varying p1, varying int32 * uniform * varying p2, varying int32 * uniform * uniform * varying p3)] function name: foo
  foo<int>(vp1, vp2, vp3);
  // CHECK: FunctionSymbolExpr {{.*}} [ void(varying int32 * varying p1, varying int32 * uniform * varying p2, varying int32 * uniform * uniform * varying p3)] function name: foo
  foo<varying int>(vp1, vp2, vp3);
  // CHECK: FunctionSymbolExpr {{.*}} [ void(uniform int32 * varying p1, uniform int32 * uniform * varying p2, uniform int32 * uniform * uniform * varying p3)] function name: foo
  foo<uniform int>(up1, up2, up3);

  // CHECK: FunctionSymbolExpr {{.*}} [ void(varying int32 * varying p1, varying int32 * uniform * varying p2, varying int32 * uniform * uniform * varying p3)] function name: foo
  foo(vp1, vp2, vp3);
  // CHECK: FunctionSymbolExpr {{.*}} [ void(uniform int32 * varying p1, uniform int32 * uniform * varying p2, uniform int32 * uniform * uniform * varying p3)] function name: foo
  foo(up1, up2, up3);
}
