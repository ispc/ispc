// The test checks that template type and non-type parameters can be used to create uniform short vector
// RUN: %{ispc}  %s --emit-llvm-text --target=host --nostdlib -o - | FileCheck %s

// CHECK-LABEL: define {{.*}} <4 x i32> @bar___vyiCuni4{{.*}}(<4 x i32> %a, <4 x i32> %b, {{.*}}
// CHECK: ret <4 x i32> %r
template <typename T, uint N> noinline uniform T<N> bar(uniform const T<N> a, uniform const T<N> b, uniform int c) {
    uniform T<N> r = {c, 2 * c, 3 * c, 0};
    for (uniform int i = 0; i < N; i++) {
        r.x += i + c;
    }
    r.y = a.x + b.z;
    return r;
}

uniform int<4> foo(uniform const int<4> a, uniform const int<4> b, uniform int c) { return bar<int, 4>(a, b, c); }