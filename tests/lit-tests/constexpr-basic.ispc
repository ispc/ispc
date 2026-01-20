// RUN: %{ispc} %s --target=host --nostdlib --emit-llvm-text -o - | FileCheck %s

constexpr uniform int A = 7;
constexpr uniform int B = A * 2;

constexpr uniform int add(uniform int x, uniform int y) { return x + y; }

uniform int g[B];
uniform int h[add(3, 4)];

// CHECK: @g = {{.*}} [14 x i32]
// CHECK: @h = {{.*}} [7 x i32]
