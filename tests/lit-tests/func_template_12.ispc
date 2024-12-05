// Test that we can mix and match regular functions and function templates with the same name.
// RUN: %{ispc}  %s --emit-llvm-text --target=host --nostdlib -o - | FileCheck %s

// CHECK-LABEL: define <4 x i32> @foo___
// CHECK: call <4 x i32> @max_func___uniCuni4
// CHECK-LABEL: define linkonce_odr <4 x i32> @max_func___uniCuni4
// CHECK: call <{{.*}} x i32> @max_func___vyivyi

uniform int max_func(uniform int a, uniform int b);
varying int max_func(varying int a, varying int b);

template <typename T, uint N>
noinline T<N> max_func(T<N> a, T<N> b) {
    T<N> r;
    foreach (i = 0 ... N) {
        r[i] = max_func(a[i], b[i]);
    }
    return r;
}
uniform int<4> foo(uniform int<4> a, uniform int<4> b) {
    return max_func<uniform int, 4>(a, b);
}