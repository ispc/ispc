// This test is checking that template type and non-type parameters
// can be used for short vectors declaration.
// RUN: %{ispc} %s --target=host 2>&1 | FileCheck %s
// CHECK-NOT: Error:
template <typename T, uint N>
uniform T<N> max_test(uniform const T<N> a, uniform const T<N> b)
{
    uniform T<N> r;
    foreach (i = 0 ... N) {
        r[i] = max(a[i], b[i]);
    }
    return r;
}

uniform double<4> foo(uniform const double<4> a, uniform const double<4> b) {
    return max_test<double, 4>(a, b);
}