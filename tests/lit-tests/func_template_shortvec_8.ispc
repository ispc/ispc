// Check compiler diagnostics
// RUN: not %{ispc} --target=host %s --nowrap 2>&1 | FileCheck %s
// CHECK: Error: Undeclared symbol "M". Did you mean "N"?
// CHECK: Error: syntax error, unexpected TOKEN_TYPE_NAME.

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
