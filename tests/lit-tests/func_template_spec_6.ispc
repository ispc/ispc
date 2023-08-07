// Test function template specialization with nested templates.

// RUN: %{ispc}  %s --emit-llvm-text --target=host -O0 -o - | FileCheck %s

float k;
template <typename T> noinline T add(T A, T B) { return A + B; }
template <typename T> noinline T mul(T A, T B) { return A * B; }
template <typename T> noinline T fma(T A, T B, T C) { return add<T>(mul<T>(A, B), C); }
template <> noinline float fma<float>(float A, float B, float C) {
    return add<float>(mul<float>(A, B), C * k);
}

// CHECK: define {{.*}} @fma___vyf___vyfvyfvyf
// CHECK: call {{.*}} @mul___vyf___vyfvyf(<{{[0-9]*}} x float>
// CHECK: fmul
// CHECK: call {{.*}} @add___vyf___vyfvyf(<{{[0-9]*}} x float>

// CHECK: define {{.*}} @mul___vyf___vyfvyf

// CHECK: define {{.*}} @add___vyf___vyfvyf

// CHECK: define <{{[0-9]*}} x float> @foo___vyfvyfvyf
// CHECK: call {{.*}} @fma___vyf___vyfvyfvyf(<{{[0-9]*}} x float>

// CHECK: define <{{[0-9]*}} x i32> @foo___vyivyivyi
// CHECK: call {{.*}} @fma___vyi___vyivyivyi(<{{[0-9]*}} x i32>

// CHECK: define {{.*}} @fma___vyi___vyivyivyi
// CHECK: call {{.*}} @mul___vyi___vyivyi(<{{[0-9]*}} x i32>
// CHECK-NOT: fmul
// CHECK: call {{.*}} @add___vyi___vyivyi(<{{[0-9]*}} x i32>

// CHECK: define {{.*}} @mul___vyi___vyivyi

// CHECK: define {{.*}} @add___vyi___vyivyi

float foo(float A, float B, float C) { return fma<varying float>(A, B, C); }
int foo(int A, int B, int C) { return fma<varying int>(A, B, C); }
