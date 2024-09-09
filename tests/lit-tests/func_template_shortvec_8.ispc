// Check compiler diagnostics
// RUN: not %{ispc} --target=host %s --nowrap 2>&1 | FileCheck %s
// CHECK: 10:38: Error: Unknown identifier "M" is used to specify the size of a vector type.
// CHECK: 11:7: Error: syntax error, unexpected TOKEN_TYPE_NAME.

struct S {
    int x;
};
template<typename T, int N>
T CalculateDifferentWidths(T<N> x, T<M> y) {
    T<S> soaValue;
    foreach (i = 0 ... N) {
        soaValue[i] = x[i] + y[i];
    }
    return soaValue[2];
}

int test_CalculateDifferentWidths(int<4> x, int<4> y) {
    return CalculateDifferentWidths<int, 4>(x, y);
}