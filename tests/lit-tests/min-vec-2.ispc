// This test checks that proper min functions is called for short vectors templated call

// RUN: %{ispc} --ast-dump --target=host --nowrap %s -o %t.o --debug-phase=10:10 2>&1 | FileCheck %s

// CHECK-LABEL: Function{{.*}}foo
// CHECK-NEXT:  |-(param 0) [uniform int32<4>] foo_a
// CHECK-NEXT:  |-(param 1) [uniform int32<4>] foo_b
// CHECK-NEXT:  `-(body) StmtList {{.*}}
// CHECK-NEXT:    `-ReturnStmt {{.*}}
// CHECK-NEXT:      `-FunctionCallExpr {{.*}} [uniform int32<4>]
// CHECK-NEXT:        |-(func) FunctionSymbolExpr {{.*}} [ uniform int32<4>(uniform int32<4> a, uniform int32<4> b)] function name: min
// CHECK-NEXT:        `-(args) ExprList {{.*}}
// CHECK-NEXT:          |-SymbolExpr {{.*}} [uniform int32<4>] symbol name: foo_a
// CHECK-NEXT:          `-SymbolExpr {{.*}} [uniform int32<4>] symbol name: foo_b
uniform int<4> foo(uniform int<4> foo_a, uniform int<4> foo_b) {
  return min(foo_a, foo_b);
}

// CHECK-LABEL: define {{.*}} @foo___uni_3C_4_3E_uni_3C_4_3E_
// CHECK-DAG: call <4 x i32> @min___vyiCuni4___uni_3C_4_3E_uni_3C_4_3E_

// CHECK-LABEL: define {{.*}} @min___vyiCuni4___uni_3C_4_3E_uni_3C_4_3E_
// CHECK-DAG: call <[[TW:.*]] x i32> @min___vyivyi(<[[TW]] x i32> {{.*}}, <[[TW]] x i32> {{.*}}, <[[TW]] x {{.*}}> {{.*}})

// CHECK-LABEL: define {{.*}} @min___vyivyi
// CHECK-DAG: call <[[TW]] x i32> @__min_varying_int32(<[[TW]] x i32> {{.*}}, <[[TW]] x i32> {{.*}})
