// This test checks short vectors with template specialization
// RUN: %{ispc}  %s --emit-llvm-text --target=host --nostdlib -o - | FileCheck %s
// CHECK-LABEL: @CalculateDifferentWidths___vyiCuni4_{{.*}}([4 x <{{.*}} x i32>] %x, [4 x <{{.*}} x i32>] %y
// CHECK-LABEL: @test_CalculateDifferentWidths
// CHECK-LABEL: @CalculateDifferentWidths___vyfCuni3_{{.*}}([3 x <{{.*}} x float>] %x, [3 x <{{.*}} x float>] %y
// CHECK-LABEL: @CalculateDifferentWidths___vyiCuni3_{{.*}}([3 x <{{.*}} x i32>] %x, [3 x <{{.*}} x i32>] %y
template<typename T, int N>
noinline T CalculateDifferentWidths(T<N> x, T<N> y) {
    T<N> soaValue;
    foreach (i = 0 ... N) {
        soaValue[i] = x[i] + y[i];
    }
    return soaValue[2];
}

template <> noinline int CalculateDifferentWidths<int, 4>(int<4> x, int<4> y) {
    int<3> soaValue;
    foreach (i = 0 ... 3) {
        soaValue[i] = x[i] + y[i];
    }
    return soaValue[2];
}

int test_CalculateDifferentWidths(uniform int c) {
    uniform int<4> x4 = {c, 2 * c, 3 * c, 0};
    uniform int<4> y4 = {3* c, 2 * c, c, 0};
    uniform float<3> x3 = {c, 2 * c, 3 * c};
    uniform float<3> y3 = {3* c, 2 * c, c};
    return CalculateDifferentWidths<int, 4>(x4, y4) + // specialization
           CalculateDifferentWidths<float, 3>(x3, y3) + // instantiation
           CalculateDifferentWidths<int, 3>(x3, y3); // instantiation
}