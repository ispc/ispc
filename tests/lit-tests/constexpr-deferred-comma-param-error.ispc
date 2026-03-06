// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -o /dev/null 2>&1 | FileCheck %s

constexpr uniform int f();
void bad(uniform int x = (f(), 1)) {}

// CHECK: Error: Default value for parameter "x" must be a compile-time constant.
