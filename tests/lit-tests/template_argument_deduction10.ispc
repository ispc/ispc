// Check here the case when user incorrectly passes vector of different sizes to tempalate function

// RUN: not %{ispc} -O2 --target=host --nowrap --nostdlib --emit-llvm-text %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Unable to find any matching overload for call to function "add".

template <typename T, uint N> uniform T<N> add(uniform T<N> a, uniform T<N> b) {
  return a + b;
}

uniform int<4> bar(uniform int<4> a, uniform int<3> b) {
  return add(a, b);
}
