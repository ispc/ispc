// Basic test to instantiate template with integer parameters.
// RUN: %{ispc}  %s --emit-llvm-text --target=host --nostdlib -o - | FileCheck %s

// CHECK-LABEL: define <{{[0-9]*}} x i32> @foo
// CHECK: call <{{[0-9]*}} x i32> @sum___vyiCuni2Cuni4___
// CHECK: call <{{[0-9]*}} x i32> @sum___vyiCuni3Cuni5___
// CHECK: call <{{[0-9]*}} x i32> @sum___Cuni2Cuni4___
// CHECK: call <{{[0-9]*}} x i32> @sum___Cuni3Cuni5___

// CHECK-LABEL: define linkonce_odr <{{[0-9]*}} x i32> @sum___vyiCuni2Cuni4___
// CHECK: ret {{.*}} i32 6
// CHECK-LABEL: define linkonce_odr <{{[0-9]*}} x i32> @sum___vyiCuni3Cuni5___
// CHECK: ret {{.*}} i32 8

// CHECK-LABEL: define linkonce_odr <{{[0-9]*}} x i32> @sum___Cuni2Cuni4___
// CHECK: ret {{.*}} i32 7
// CHECK-LABEL: define linkonce_odr <{{[0-9]*}} x i32> @sum___Cuni3Cuni5___
// CHECK: ret {{.*}} i32 9

template<typename T, int x, int y>
noinline T sum() {
    return x + y;
}

template<int x, int y>
noinline int sum() {
    return x + y + 1;
}

int foo() {
    return sum<int, 2, 4>() + sum<int, 3, 5>() + sum<2, 4>() + sum<3, 5>();
}