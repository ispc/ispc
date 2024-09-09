// Check that short vector declared using template parameters uses DependentMemberExpr in AST

// RUN: %{ispc} %s --nostdlib --target=host --ast-dump -o %t.o | FileCheck %s
// CHECK-LABEL: "test1"
// CHECK: AssignExpr {{.*}} [uniform <dependent type>], '='
// CHECK: IndexExpr {{.*}} [uniform <dependent type>]
// CHECK: AssignExpr {{.*}} [uniform <dependent type>], '='
// CHECK: DependentMemberExpr {{.*}} [uniform <dependent type>] . zyx
// CHECK-NOT: NULL

// CHECK-LABEL: "test2"
// CHECK: AssignExpr {{.*}} [uniform <dependent type>], '='
// CHECK: IndexExpr {{.*}} [uniform <dependent type>]
// CHECK: AssignExpr {{.*}} [uniform <dependent type>], '='
// CHECK: DependentMemberExpr {{.*}} [uniform <dependent type>] . zyx
// CHECK-NOT: NULL

// CHECK-LABEL: "test3"
// CHECK: AssignExpr {{.*}} [uniform <dependent type>], '='
// CHECK: IndexExpr {{.*}} [uniform <dependent type>]
// CHECK: AssignExpr {{.*}} [uniform <dependent type>], '='
// CHECK: DependentMemberExpr {{.*}} [uniform <dependent type>] . zyx
// CHECK-NOT: NULL

template<typename T>
T test1(T<3> x, T<3> y, uniform int N) {
    T<3> val;
    foreach (i = 0 ... N) {
        val[i] = x[i] + y[i];
    }
    val = val.zyx;
    return val[2];
}

template<uint N>
int test2(int<N> x, int<N> y) {
    int<N> val;
    foreach (i = 0 ... N) {
        val[i] = x[i] + y[i];
    }
    val = val.zyx;
    return val[2];
}

template<typename T, uint N>
int test3(T<N> x, T<N> y) {
    T<N> val;
    foreach (i = 0 ... N) {
        val[i] = x[i] + y[i];
    }
    val = val.zyx;
    return val[2];
}