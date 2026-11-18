// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -o %t.o 2>&1 | FileCheck %s

template <int N>
uniform int leaf() { return N; }

uniform int runtime(uniform int n) { return n + 1; }

template <int N>
uniform int non_constant() {
    // CHECK: Error: Only integral, bool, enum types and non-type template parameters are allowed as template arguments.
    // CHECK-NEXT: return leaf<runtime(N)>();
    return leaf<runtime(N)>();
}

template <int N>
uniform int non_integral() {
    // CHECK: Error: Only integral, bool, enum types and non-type template parameters are allowed as template arguments.
    // CHECK-NEXT: return leaf<N + 0.5f>();
    return leaf<N + 0.5f>();
}

uniform int test_non_constant() { return non_constant<0>(); }
uniform int test_non_integral() { return non_integral<0>(); }
