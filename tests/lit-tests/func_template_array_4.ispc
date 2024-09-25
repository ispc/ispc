// This test checks function overloading when arrays are used as parameter
// RUN: %{ispc} %s --emit-llvm-text --target=host --nostdlib -o - | FileCheck %s
// CHECK-LABEL: define void @func__
// CHECK-LABEL: define void @test___
// CHECK: call void @func___vyfCuni64Cuni32
// CHECK-LABEL: define linkonce_odr void @func___vyfCuni64Cuni32

noinline void func(uniform float arr1[64], uniform float arr2[32]) {
    uniform float O[64] = {0};
    arr1[programIndex] = O[programIndex] + arr2[programIndex];
}

template <typename T, int C1, int C2> noinline void func(uniform T arr1[C1], uniform T arr2[C2]) {
    uniform T O[C1] = {0};
    arr2[programIndex] = O[programIndex] + arr2[programIndex];
}

void test(uniform float x[], uniform float y[]) {
    func<float, 64, 32>(x, y);
}