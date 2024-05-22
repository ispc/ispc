// Tests nested templates with non-type template parameters.

// RUN: %{ispc}  %s --emit-llvm-text --target=host -O0

// CHECK-LABEL: define {{.*}} @foo___vyfvyfvyf
// CHECK: call <{{[0-9]*}} x float> @fma___vyfCuni4Cuni8___vyfvyfvyf

// CHECK-LABEL: define {{.*}} @fma___vyfCuni4Cuni8___vyfvyfvyf
// CHECK: call <{{[0-9]*}} x float> @mul___vyfCuniMCuni8___vyfvyf
// CHECK: call <{{[0-9]*}} x float> @add___vyfCuniNCuni4___vyfvyf
// CHECK-LABEL: define {{.*}} @mul___vyfCuniMCuni8___vyfvyf
// CHECK-LABEL: define {{.*}} @add___vyfCuniNCuni4___vyfvyf

template <typename T, int N> noinline T add(T A, T B) { return A + B + N; }
template <typename T, int M> noinline T mul(T A, T B) { return A * B + M; }
template <typename T, int N, int M> noinline T fma(T A, T B, T C) { return add<T, N>(mul<T, M>(A, N), C); }

float foo(float A, float B, float C) { return fma<varying float, 4, 8>(A, B, C); }