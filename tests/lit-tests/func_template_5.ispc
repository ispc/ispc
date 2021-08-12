// Tests negative case for errors - incorrect typename types or number of
// parameters.

//; RUN: not %{ispc} %s -o %t.o --nowrap --target=host 2>&1 | FileCheck %s

//; CHECK: 12:48: Error: Unable to find any matching overload for call to function template "add2".

//; CHECK: 14:39: Error: Unable to find any matching overload for call to function template "add2".

template <typename T> noinline T add2(T A, T B) { return A + B; }

float foo0(float A, float B, float C) { return add2<varying float>(A, B, C); }

float foo1(float A, float B) { return add2<varying float, int>(A, B); }
