// Tests nested templates.

// RUN: %{ispc}  %s --emit-llvm-text --target=host -O0 -o - | FileCheck %s

template <typename T> noinline T add(T A, T B) { return A + B; }
template <typename T> noinline T mul(T A, T B) { return A * B; }
template <typename T> noinline T fma(T A, T B, T C) { return add<T>(mul<T>(A, B), C); }

// CHECK: define <{{[0-9]*}} x float> @foo___vyfvyfvyf
// CHECK: call <{{[0-9]*}} x float> @fmavyf___vyfvyfvyf(<{{[0-9]*}} x float>

// CHECK: define <{{[0-9]*}} x float> @fmavyf___vyfvyfvyf
// CHECK: call <{{[0-9]*}} x float> @mulvyf___vyfvyf(<{{[0-9]*}} x float>
// CHECK: call <{{[0-9]*}} x float> @addvyf___vyfvyf(<{{[0-9]*}} x float>

// CHECK: define <{{[0-9]*}} x float> @mulvyf___vyfvyf

// CHECK: define <{{[0-9]*}} x float> @addvyf___vyfvyf
float foo(float A, float B, float C) { return fma<varying float>(A, B, C); }
