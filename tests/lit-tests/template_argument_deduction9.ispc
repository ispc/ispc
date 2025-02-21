// Both uniform and varying functions are instantiated, and they are properly used.

// RUN: %{ispc} %s --nowrap --nostdlib --target=host --ast-dump -o %t.o 2>&1 | FileCheck %s

// CHECK: (instantiation <uniform int32, 4>) Function {{.*}} [ uniform int32<4>(uniform int32<4> a)] "bar1"
// CHECK: (instantiation <varying int32, 4>) Function {{.*}} [ varying int32<4>(varying int32<4> a)] "bar1"
template <typename T, uint N> T<N> bar1(T<N> a) { return a; }

// CHECK:      |-Function {{.*}}  [ uniform int32<4>(uniform int32<4> x)] "func1_u"
// CHECK-NEXT: | |-(param 0) [uniform int32<4>] x
// CHECK-NEXT: | `-(body) StmtList {{.*}}
// CHECK-NEXT: |   `-ReturnStmt {{.*}}
// CHECK-NEXT: |     `-FunctionCallExpr {{.*}} [uniform int32<4>]
// CHECK-NEXT: |       |-(func) FunctionSymbolExpr {{.*}} [ uniform int32<4>(uniform int32<4> a)] function name: bar1
// CHECK-NEXT: |       `-(args) ExprList {{.*}}
// CHECK-NEXT: |         `-SymbolExpr {{.*}} [uniform int32<4>] symbol name: x
uniform int<4> func1_u(uniform int<4> x) { return bar1(x); }

// CHECK:      |-Function {{.*}}  [ varying int32<4>(varying int32<4> x)] "func1_v"
// CHECK-NEXT: | |-(param 0) [varying int32<4>] x
// CHECK-NEXT: | `-(body) StmtList {{.*}}
// CHECK-NEXT: |   `-ReturnStmt {{.*}}
// CHECK-NEXT: |     `-FunctionCallExpr {{.*}} [varying int32<4>]
// CHECK-NEXT: |       |-(func) FunctionSymbolExpr {{.*}} [ varying int32<4>(varying int32<4> a)] function name: bar1
// CHECK-NEXT: |       `-(args) ExprList {{.*}}
// CHECK-NEXT: |         `-SymbolExpr {{.*}} [varying int32<4>] symbol name: x
varying int<4> func1_v(varying int<4> x) { return bar1(x); }

// CHECK:      `-Function {{.*}}  [ varying int32<4>(varying int32<4> x)] "func1"
// CHECK-NEXT:   |-(param 0) [varying int32<4>] x
// CHECK-NEXT:   `-(body) StmtList {{.*}}
// CHECK-NEXT:     `-ReturnStmt {{.*}}
// CHECK-NEXT:       `-FunctionCallExpr {{.*}} [varying int32<4>]
// CHECK-NEXT:         |-(func) FunctionSymbolExpr {{.*}} [ varying int32<4>(varying int32<4> a)] function name: bar1
// CHECK-NEXT:         `-(args) ExprList {{.*}}
// CHECK-NEXT:           `-SymbolExpr {{.*}} [varying int32<4>] symbol name: x
int<4> func1(int<4> x) { return bar1(x); }
