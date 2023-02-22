// Tests nested templates.

// RUN: %{ispc}  %s --emit-llvm-text --target=host -O0 -o - | FileCheck %s

template <typename T> noinline T add(T A, T B) { return A + B; }
template <typename T> noinline T mul(T A, T B) { return A * B; }
template <typename T> noinline T fma(T A, T B, T C) { return add<T>(mul<T>(A, B), C); }

// CHECK: define <{{[0-9]*}} x float> @foo___vyfvyfvyf
// CHECK: call {{.*}} @fma___vyf___vyfvyfvyf(<{{[0-9]*}} x float>

// CHECK: define {{.*}} @fma___vyf___vyfvyfvyf
// CHECK: call {{.*}} @mul___vyf___vyfvyf(<{{[0-9]*}} x float>
// CHECK: call {{.*}} @add___vyf___vyfvyf(<{{[0-9]*}} x float>

// CHECK: define {{.*}} @mul___vyf___vyfvyf

// CHECK: define {{.*}} @add___vyf___vyfvyf
float foo(float A, float B, float C) { return fma<varying float>(A, B, C); }
