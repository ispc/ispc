// Check short vectors in template argument deduction. More different cases.
// RUN: %{ispc} %s --nowrap --nostdlib --target=host --ast-dump -o %t.o 2>&1 | FileCheck %s

// CHECK: (instantiation <uniform int32, 4, varying int64>) Function {{.*}}  [ varying int64(uniform int32<4> a, varying int64<4> b, varying int64 x)] "foo"
template <typename T, uint N, typename G> int64 foo(T<N> a, G<N> b, G x) { return a[0] + b[0] + x; }
int64 boo(uniform int<4> x, uniform int64<4> y, int64 z) { return foo(x, y, z); }

// CHECK: (instantiation <uniform int32, 4, 2>) Function {{.*}}  [ varying int64(uniform int32<4> a, uniform int32<2> b)] "func"
template <typename T, uint N, int W> int64 func(T<N> a, T<W> b) { return a[0] + b[0] + N + W; }
int64 func_call(uniform int<4> x, uniform int<2> y) { return func(x, y); }

// CHECK: (instantiation <uniform int32, uniform int64, 7, 2>) Function {{.*}}  [ varying int64(uniform int32<2> a, uniform int64<7> b)] "bar"
template <typename T, typename G, int W, int H> int64 bar(T<H> a, G<W> b) { return a.x + b.x; }
int64 baz(uniform int<2> x, uniform int64<7> y) { return bar(x, y); }
