// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -o /dev/null 2>&1 | FileCheck %s

constexpr uniform int f();
constexpr uniform int bad = (f(), 1);

// CHECK: Error: Initializer for global variable "bad" must be a constant.
