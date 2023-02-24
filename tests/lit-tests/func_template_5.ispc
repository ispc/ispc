// Tests negative case for errors - incorrect typename types or number of parameters.

// RUN: not %{ispc} %s -o %t.o --nowrap --target=host 2>&1 | FileCheck %s

template <typename T> noinline T add(T A, T B) { return A + B; }

// CHECK: 8:40: Error: Unable to find any matching overload for call to function "add".
void foo0(float A, float B, float C) { add<varying float>(A, B, C); }

// CHECK: 11:31: Error: Unable to find any matching overload for call to function "add".
void foo1(float A, float B) { add<varying float, int>(A, B); }

