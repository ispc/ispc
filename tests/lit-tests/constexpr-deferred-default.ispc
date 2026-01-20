// RUN: %{ispc} %s --target=host --nostdlib --emit-llvm-text -O0 -o - | FileCheck %s

constexpr uniform int later();

uniform int bar(uniform int x = later()) { return x + 1; }

constexpr uniform int later() { return 6; }

uniform int use() { return bar(); }

// CHECK-LABEL: @use
// CHECK: store i32 6
