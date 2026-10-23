// Check that enums declared in different function bodies with the same name
// are different types with their own template instantiations, which get
// internal linkage, as for local structs (see 3959.ispc and 3962.ispc).
// Global enums keep their mangled names and linkonce_odr linkage.  Global
// anonymous enums are mangled by their first enumerator, which is the same in
// each translation unit, and template instantiations over them, including
// over non-type template parameters of their types, get internal linkage.
// A value of a local or anonymous enum passed to an int parameter is converted
// to int, so the instantiation keeps linkonce_odr linkage and can be merged
// with the same instantiation from other translation units.  Enums are distinguished by identity, not by source
// position, so two enums whose names have the same position (e.g. after #line
// directives or when a header is included twice) are different types.

// RUN: %{ispc} %s --emit-llvm-text --target=host --nostdlib --nowrap -O0 -o - | FileCheck %s

template <typename E> noinline uniform int conv(uniform E e) { return (uniform int)e; }

// CHECK-DAG: define internal i32 @conv___vyenum_5B_LE.1_5D____unenum_5B_LE.1_5D_(
uniform int f1() {
    enum LE { A = 1 };
    uniform LE e = A;
    return conv(e);
}

// CHECK-DAG: define internal i32 @conv___vyenum_5B_LE.2_5D____unenum_5B_LE.2_5D_(
uniform int f2() {
    enum LE { B = 2 };
    uniform LE e = B;
    return conv(e);
}

// CHECK-DAG: define internal i32 @conv___vyenum_5B_.1_5D____unenum_5B_.1_5D_(
uniform int f3() {
    uniform enum { C = 3 } e = C;
    return conv(e);
}

// CHECK-DAG: define linkonce_odr i32 @conv___vyenum_5B_G_5D____unenum_5B_G_5D_(
enum G { X = 4 };
uniform int f4() {
    uniform G g = X;
    return conv(g);
}

// CHECK-DAG: define i32 @h___unenum_5B_P_5D_(
// CHECK-DAG: define i32 @h___unenum_5B_Q_5D_(
#line 100
enum P { P0 };
#line 100
enum Q { Q0 };
uniform int h(uniform P p) { return 1; }
uniform int h(uniform Q q) { return 2; }

// CHECK-DAG: define internal i32 @conv___vyenum_5B_.GA_5D____unenum_5B_.GA_5D_(
// CHECK-DAG: define internal i32 @conv___vyenum_5B_.GB_5D____unenum_5B_.GB_5D_(
// CHECK-DAG: define i32 @k___unenum_5B_.GA_5D_(
#line 200
typedef enum { GA = 1 } GAE;
#line 200
typedef enum { GB = 2 } GBE;
uniform int k(uniform GAE a) {
    uniform GBE b = GB;
    return conv(a) + conv(b);
}

// CHECK-DAG: define internal i32 @nt___Cunenum_5B_.GA_5D_1___(
template <GAE V> noinline uniform int nt() { return V; }
uniform int n() { return nt<GA>(); }

// CHECK-DAG: define linkonce_odr i32 @ni___Cunenum_5B_.GA_5D_1___(
// CHECK-DAG: define linkonce_odr i32 @ni___Cunenum_5B_LE.3_5D_2___(
template <int N> noinline uniform int ni() { return N; }
uniform int ni1() { return ni<GA>(); }
uniform int ni2() {
    enum LE { D = 2 };
    return ni<D>();
}
