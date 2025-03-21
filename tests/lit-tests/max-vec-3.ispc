// This test checks that proper max functions is called for short vectors templated call

// RUN: %{ispc} --ast-dump --target=host --nowrap %s -o %t.o --debug-phase=10:10 2>&1 | FileCheck %s

// CHECK-LABEL: Function{{.*}}foo
// CHECK-NEXT:  |-(param 0) [varying int32<4>] foo_a
// CHECK-NEXT:  |-(param 1) [varying int32<4>] foo_b
// CHECK-NEXT:  `-(body) StmtList {{.*}}
// CHECK-NEXT:    `-ReturnStmt {{.*}}
// CHECK-NEXT:      `-FunctionCallExpr {{.*}} [varying int32<4>]
// CHECK-NEXT:        |-(func) FunctionSymbolExpr {{.*}} [ varying int32<4>(varying int32<4> a, varying int32<4> b)] function name: max
// CHECK-NEXT:        `-(args) ExprList {{.*}}
// CHECK-NEXT:          |-SymbolExpr {{.*}} [varying int32<4>] symbol name: foo_a
// CHECK-NEXT:          `-SymbolExpr {{.*}} [varying int32<4>] symbol name: foo_b
varying int<4> foo(varying int<4> foo_a, varying int<4> foo_b) {
  return max(foo_a, foo_b);
}

// CHECK-LABEL: define {{.*}} @foo___vyi_3C_4_3E_vyi_3C_4_3E_
// CHECK-DAG: call [4 x <[[TW:.*]] x i32>] @max___uniCuni4___vyi_3C_4_3E_vyi_3C_4_3E_

// CHECK-LABEL: define {{.*}} @max___uniCuni4___vyi_3C_4_3E_vyi_3C_4_3E_
// CHECK-DAG: call <[[TW]] x i32> @max___vyivyi(<[[TW]] x i32> {{.*}}, <[[TW]] x i32> {{.*}}, <[[TW]] x {{.*}}> {{.*}})

// CHECK-LABEL: define {{.*}} @max___vyivyi
// CHECK-DAG: call <[[TW]] x i32> @__max_varying_int32(<[[TW]] x i32> {{.*}}, <[[TW]] x i32> {{.*}})
