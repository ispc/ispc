// Issue #2116 : This test ensures that using the result of
// unary expression on a reference type for other operations
// doesn't cause a crash.
// RUN: %{ispc} %s -O2 --target=host

// Case 1: 'if' in 'foo0' uses result of post inc operation 'param_inout++'
void foo0(uniform int &param_inout) {
    if (param_inout++ > 0) {
        param_inout++;
    }
}

// Case 2: 'if' in 'foo1' uses result of pre inc operation '++param_inout'
void foo1(uniform int &param_inout) {
    if (++param_inout > 0) {
        param_inout++;
    }
}
