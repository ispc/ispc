// REQUIRES: X86_ENABLED && X86_64_HOST
// RUN: %{ispc} %s --target=generic-i32x4 --nostdlib --emit-llvm-text -o - | FileCheck %s

struct Pair {
    uniform int a;
    uniform int b;
};

constexpr uniform Pair P0 = {1, 4};
constexpr uniform int PB = P0.b;

constexpr uniform int Arr0[3] = {10, 20, 30};
constexpr uniform int Arr1 = Arr0[2];

constexpr uniform int<4> make_vec(uniform int base) {
    uniform int<4> v = {base, base + 1, base + 2, base + 3};
    return v + 2;
}

constexpr uniform int<4> V0 = make_vec(3);
constexpr uniform int VY = V0.y;
constexpr uniform int<2> VXY = V0.xy;
constexpr uniform int<4> VBroadcast = 3;

constexpr uniform int arr_sum() {
    uniform int a[3] = {1, 2, 3};
    return a[1] + a[2];
}

constexpr uniform int A0 = arr_sum();

uniform int foo() { return PB + A0 + Arr1; }
uniform int<2> bar() { return VXY; }
uniform int<4> baz() { return V0; }

// CHECK-DAG: @PB = {{.*}} i32 4
// CHECK-DAG: @Arr1 = {{.*}} i32 30
// CHECK-DAG: @A0 = {{.*}} i32 5
// CHECK-DAG: @V0 = {{.*}} <4 x i32> <i32 5, i32 6, i32 7, i32 8>
// CHECK-DAG: @VY = {{.*}} i32 6
// CHECK-DAG: @VXY = {{.*}} <4 x i32> <i32 5, i32 6, i32 undef, i32 undef>
// CHECK-DAG: @VBroadcast = {{.*}} {{<4 x i32> splat \(i32 3\)|<4 x i32> <i32 3, i32 3, i32 3, i32 3>}}
