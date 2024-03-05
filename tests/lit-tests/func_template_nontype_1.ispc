// Basic test to instantiate template with integer parameters.
// RUN: %{ispc}  %s --emit-llvm-text --target=host --nostdlib -o - | FileCheck %s

// CHECK-LABEL: define <{{[0-9]*}} x i32> @foo
// CHECK: call <{{[0-9]*}} x i32> @sum___vyi24___
// CHECK: call <{{[0-9]*}} x i32> @sum___vyi35___

// CHECK-LABEL: define linkonce_odr <{{[0-9]*}} x i32> @sum___vyi24___
// CHECK-LABEL: define linkonce_odr <{{[0-9]*}} x i32> @sum___vyi35___

template<typename T, int x, int y>
noinline T sum() {
    return x + y;
}

int foo() {
    return sum<int, 2, 4>() + sum<int, 3, 5>();
}