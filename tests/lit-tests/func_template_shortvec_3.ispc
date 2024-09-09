// The test checks that enum type as template parameter can be used to create short vector
// RUN: %{ispc}  %s --emit-llvm-text --target=host --nostdlib -o - | FileCheck %s
// CHECK: <{{.*}} x i32> @bar___vyiCunenum_{{.*}}([3 x <{{.*}} x i32>] %x, [3 x <{{.*}} x i32>] %y
enum Foo {
    ZERO,
    ONE,
    TWO,
    THREE
};

template<typename T, Foo N>
noinline T bar(T<N> x, T<N> y) {
    T<N> soaValue;
    foreach (i = 0 ... N) {
        soaValue[i] = x[i] + y[i];
    }
    return soaValue[2];
}

int test(uniform int c) {
    uniform int<3> x3 = {c, 2 * c, 3 * c};
    uniform int<3> y3 = {3* c, 2 * c, c};
    return bar<varying int, THREE>(x3, y3);
}