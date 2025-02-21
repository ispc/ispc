// The test checks that enum type can be used for array indexing
// RUN: %{ispc} %s --ast-dump --target=host --nostdlib | FileCheck %s
// RUN: %{ispc} %s --target=host --nostdlib -o %t.o

// CHECK: FunctionTemplate {{.*}} [ /*unbound*/ T(varying int32 * uniform a)] "func"
// CHECK Variable arr (/*unbound*/ T[C])
// CHECK: (instantiation <varying int32, 64, 0, 3>) {{.*}}  [ varying int32(varying int32 * uniform a)] "func"
// CHECK: ReturnStmt
// CHECK: IndexExpr
// CHECK: SymbolExpr {{.*}} [varying int32[64]] symbol name: arr
// CHECK: ConstExpr {{.*}} [const uniform unsigned int64] (0)
// CHECK: IndexExpr
// CHECK: SymbolExpr {{.*}} [varying int32[64]] symbol name: arr
// CHECK: ConstExpr {{.*}} [const uniform unsigned int64] (3)

enum Foo {
    ZERO,
    ONE,
    TWO,
    THREE
};

template<typename T, uint C, Foo N1, Foo N2>
noinline T func(int a[]) {
    T arr[C];
    for (uniform uint i = 0; i < C; i++) {
        arr[i] = a[i] + i;
    }
    return arr[N1] + arr[N2];
}

int test(int a[]) {
    return func<int, 64, ZERO, THREE>(a);
}

