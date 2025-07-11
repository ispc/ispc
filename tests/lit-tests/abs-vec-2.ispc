// This test checks that proper abs functions is called for short vectors templated call

// RUN: %{ispc} --ast-dump --target=host --nowrap %s -o %t.o --debug-phase=10:10 2>&1 | FileCheck %s

#include <short_vec.isph>

// CHECK-LABEL: Function{{.*}}foo
// CHECK-NEXT:  |-(param 0) [uniform int32<4>] foo_a
// CHECK-NEXT:  `-(body) StmtList {{.*}}
// CHECK-NEXT:    `-ReturnStmt {{.*}}
// CHECK-NEXT:      `-FunctionCallExpr {{.*}} [uniform int32<4>]
// CHECK-NEXT:        |-(func) FunctionSymbolExpr {{.*}} [ uniform int32<4>(uniform int32<4> a)] function name: abs
// CHECK-NEXT:        `-(args) ExprList {{.*}}
// CHECK-NEXT:          `-SymbolExpr {{.*}} [uniform int32<4>] symbol name: foo_a
uniform int<4> foo(uniform int<4> foo_a) {
  return abs(foo_a);
}

// CHECK-LABEL: define {{.*}} @foo___uni_3C_4_3E_
// CHECK-DAG: call <4 x i32> @abs___vyiCuni4___uni_3C_4_3E_

// CHECK-LABEL: define {{.*}} @abs___vyiCuni4___uni_3C_4_3E_
// CHECK-DAG: call <[[TW:.*]] x i32> @abs___vyi(<[[TW]] x i32> {{.*}}, <[[TW]] x {{.*}}> {{.*}})
