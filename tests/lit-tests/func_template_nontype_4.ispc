// Tests nested templates with non-type template parameters.

// RUN: %{ispc}  %s --emit-llvm-text --target=host -O0 -o - | FileCheck %s

// XFAIL: *

template <typename T, int N> noinline T add(T A, T B) { return A + B + N; }
template <typename T, int M> noinline T mul(T A, T B) { return A * B + M; }
template <typename T, int N, int M> noinline T fma(T A, T B, T C) { return add<T, N>(mul<T, M>(A, B), C); }

float foo(float A, float B, float C) { return fma<varying float, 4, 8>(A, B, C); }
