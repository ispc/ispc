// The test checks that template type and non-type parameters can be used to create varying short vector
// RUN: %{ispc}  %s --emit-llvm-text --target=host --nostdlib -o - | FileCheck %s

// CHECK-LABEL: define {{.*}} [4 x <{{.*}} x i32>] @bar___vyiCuni4{{.*}}([4 x <{{.*}} x i32>] %a, [4 x <{{.*}} x i32>] %b, {{.*}}
// CHECK: ret [4 x <{{.*}} x i32>] %r
template <typename T, uint N> noinline T<N> bar(const T<N> a, const T<N> b, int c) {
    T<N> r = {c, 2 * c, 3 * c, 0};
    foreach (i = 0... N) {
        r.x += i + c;
    }
    r.y = a.x + b.z;
    return r;
}

int<4> foo(const int<4> a, const int<4> b, int c) { return bar<int, 4>(a, b, c); }