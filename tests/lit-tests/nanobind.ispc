// RUN: %{ispc} -o %t.o --nanobind-wrapper=%t.cpp %s --target=host
// RUN: FileCheck --input-file=%t.cpp %s

// CHECK: using ispc::S;

// CHECK-LABEL: int32_t foo(struct S & s, nb::ndarray<float> x, nb::ndarray<int32_t> y, int32_t count, double d, E e) {
// CHECK-NEXT:   return   ispc::foo(s, x.data(), y.data(), count, d, e);
// CHECK-NEXT: }

// CHECK-LABEL: NB_MODULE({{.*}}, m) {
// CHECK-NEXT:  nb::class_<S>(m, "S")
// CHECK-NEXT:    .def(nb::init())
// CHECK-NEXT:    .def_rw("u", &S::u)
// CHECK-NEXT:    .def_rw("unbound", &S::unbound)
// CHECK-NEXT:    .def_prop_rw("v",
// CHECK-NEXT:      make_array_getter(&S::v),
// CHECK-NEXT:      make_array_setter(&S::v))
// CHECK-NEXT:    .def_prop_rw("a",
// CHECK-NEXT:      make_array_getter(&S::a),
// CHECK-NEXT:      make_array_setter(&S::a))
// CHECK-NEXT:    .def_prop_rw("b",
// CHECK-NEXT:      make_array_getter(&S::b),
// CHECK-NEXT:      make_array_setter(&S::b))
// CHECK-NEXT:    .def_prop_rw("c",
// CHECK-NEXT:      make_array_getter(&S::c),
// CHECK-NEXT:      make_array_setter(&S::c))
// CHECK-NEXT:  ;
// CHECK-EMPTY:
// CHECK-NEXT:  nb::enum_<E>(m, "E")
// CHECK-NEXT:    .value("E1", E::E1)
// CHECK-NEXT:    .value("E2", E::E2)
// CHECK-NEXT:    .value("E3", E::E3)
// CHECK-NEXT:  ;
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT:  // Register wrappers
// CHECK-NEXT:  m.def("foo", &foo);
// CHECK-NEXT:}

struct S {
    uniform int u;
    int unbound;
    varying int v;
    uniform int a[10];
    uniform int b[5][6];
    uniform int c[2][3][4];
};

enum E {
    E1 = 1,
    E2 = 2,
    E3 = 3
};

export uniform int foo(uniform S &s, uniform float x[], uniform int y[], uniform int count, uniform double d,
                       uniform E e) {
    return 0;
}
