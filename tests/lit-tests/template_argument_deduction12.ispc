// Check here deduction for short vectors in nested templates

// RUN: %{ispc} %s --nowrap --nostdlib --target=host --ast-dump -o %t.o 2>&1 | FileCheck %s

template <typename T, uint N> T<N> same(T<N> a) { return a; }
template <typename T, uint N> T<N> inc(T<N> a) { return same(a) + 1; }

// CHECK: (instantiation <uniform int32, 4>) Function {{.*}}  [ uniform int32<4>(uniform int32<4> a)] "same"
// CHECK: (instantiation <uniform int32, 4>) Function {{.*}}  [ uniform int32<4>(uniform int32<4> a)] "inc"
uniform int<4> foo(uniform int<4> a) { return inc(a); }
