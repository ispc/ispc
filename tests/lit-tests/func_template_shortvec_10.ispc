// The test checks that we can do casts to pointers to short vectors inside templates (#3073).
// RUN: %{ispc} %s --ast-dump --target=host --nostdlib | FileCheck %s
// CHECK: unmasked void(uniform int8 * uniform a)] "foo1"
// CHECK: TypeCastExpr {{.*}} [/*unbound*/ T<4> * /*unbound*/]
// CHECK: (instantiation <varying int32>) {{.*}}  [ unmasked void(uniform int8 * uniform a)] "foo1"
// CHECK: TypeCastExpr {{.*}} [varying int32<4> * varying]

// CHECK: unmasked void(uniform int8 * uniform a)] "foo2"
// CHECK: TypeCastExpr {{.*}} [/*unbound*/ T<N> * /*unbound*/]
// CHECK: (instantiation <varying int32, 4>) {{.*}}  [ unmasked void(uniform int8 * uniform a)] "foo2"
// CHECK: TypeCastExpr {{.*}} [varying int32<4> * varying]
template <typename T> noinline unmasked void foo1(uniform int8 *uniform a) {
    uniform T<4> *uniform ptr1 = (uniform int<4> *uniform)a;
    T<4> *ptr2 = (T<4> *)a;
}
template <typename T, int N> noinline unmasked void foo2(uniform int8 *uniform a) {
    uniform T<N> *uniform ptr1 = (uniform T<N> * uniform) a;
    T<N> *ptr2 = (T<N> *)a;
}

unmasked int<4> bar(uniform int8 *uniform a) {
    foo1<int>(a);
    foo2<int, 4>(a);
}