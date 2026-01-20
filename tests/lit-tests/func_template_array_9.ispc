// Check that custom variable can't be used as template argument (yet).

// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s
// CHECK: Only integral, bool, enum types and non-type template parameters are allowed as template arguments.

template <int C> noinline void func(float arr[C]) {
    uniform float O[C] = {0};
    arr[programIndex] = O[programIndex];
}

void test(float x[]) {
    uniform int N = 32;
    func<N>(x);
}
