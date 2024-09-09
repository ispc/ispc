// Check compiler diagnostics
// RUN: not %{ispc} --target=host %s --nowrap 2>&1 | FileCheck %s
// CHECK: 14:10: Error: Only atomic types (int, float, ...) and template type parameters are legal for vector types.
// CHECK: 28:14: Error: Only atomic types (int, float, ...) and template type parameters are legal for vector types.
// CHECK: 29:14: Error: Illegal to specify vector size of 0.
// CHECK: 19:21: Warning: Array index "2" may be out of bounds for 1 element array.

struct S {
    int x;
};
template<typename T, int N>
T CalculateDifferentWidths(T<N> x, T<N> y) {
    uniform int Q = 10;
    T<Q> soaValueQ;
    T<N> soaValue;
    foreach (i = 0 ... N) {
        soaValue[i] = x[i] + y[i];
    }
    return soaValue[2];
}

int test_CalculateDifferentWidths(int<4> x, int<4> y) {
    return CalculateDifferentWidths<int, 1>(x, y);
}

float CalculateDifferentWidths(float c) {
    uniform int Q = 10;
    float<Q> soaValueQ;
    float<0> soaValue;
    foreach (i = 0 ... Q) {
        soaValue[i] = c*i;
    }
    return soaValue[2];
}