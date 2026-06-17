// RUN: %{ispc} %s --target=host --nostdlib --emit-llvm-text -O0 -o - | FileCheck %s

noinline constexpr uniform int f(uniform int x) { return x + 1; }
static noinline constexpr uniform int fs(uniform int x) { return x + 2; }

uniform int use(uniform int y) { return f(y) + fs(y); }

// CHECK: define linkonce_odr {{.*}} @f___{{.*}}(
// CHECK: define internal {{.*}} @fs___{{.*}}(
// CHECK-LABEL: @use___
// CHECK: call {{.*}} @f___{{.*}}(
// CHECK: call {{.*}} @fs___{{.*}}(
