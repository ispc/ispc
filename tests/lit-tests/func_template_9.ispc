// Check that explicit instantiation picks the right version of the template.

// RUN: %{ispc} %s --nostdlib --target=host --ast-dump -o %t.o | FileCheck %s


template <typename T> T foo(T a) { return a; }
template <typename T> T foo(T a, T b) { return a+b; }
template <typename T> T foo(T a, int b) { return a+b; }

template <typename T> T goo(T a) { return a; }
template <typename T> T goo(T a, T b) { return a+b; }
template <typename T> T goo(T a, int b) { return a+b; }

// CHECK: (instantiation <varying float>) Function {{.*}} [ varying float(varying float a, varying float b)] "foo"
template float foo<float>(float, float);

// CHECK: (instantiation <varying float>) Function {{.*}} [ varying float(varying float a, varying int32 b)] "goo"
template float goo<float>(float, int);

