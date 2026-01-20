// RUN: %{ispc} %s --target=host --nostdlib --emit-llvm-text -o - | FileCheck %s

constexpr uniform int later();

constexpr uniform int G = later();
uniform int H = later() + 5;

constexpr uniform int later() { return 12; }

uniform int use() { return G + H; }

// CHECK-DAG: @G = {{.*}} i32 12
// CHECK-DAG: @H = {{.*}} i32 17
