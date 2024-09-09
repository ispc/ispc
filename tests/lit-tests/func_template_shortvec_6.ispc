// This test checks short vectors with nested templates
// RUN: %{ispc}  %s --emit-llvm-text --target=host --nostdlib -o - | FileCheck %s
// CHECK-LABEL: <4 x i32> @test___uni
// CHECK-LABEL: <4 x i32> @bar___vyiCuni4___uni
// CHECK-LABEL: <4 x i32> @foo___vyiCuni4___uni
// Nested template
template <typename T, int N> noinline uniform T<N> foo(uniform T c) {
    uniform T<N> r = { c, 2*c, 3*c, 0};
    return r;
}

template <typename T, int N> noinline uniform T<N> bar(uniform T c) {
    return foo<T, N>(c);
}

uniform int<4> test(uniform int c) { return bar<varying int, 4>(c); }